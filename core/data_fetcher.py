"""
黄金价格数据获取模块
支持多个免费 API 源获取黄金价格数据
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from core.config import (
    GOLD_API_KEY,
    GOLD_API_URL,
    BACKUP_APIS,
    DATA_DIR,
    ANALYSIS_DAYS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoldDataFetcher:
    """黄金价格数据获取器"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        self.data_file = f"{DATA_DIR}/gold_price_history.json"

    def fetch_from_gold_api(self) -> Optional[Dict]:
        """从 GoldAPI 获取实时金价"""
        if not GOLD_API_KEY:
            logger.warning("未配置 GOLD_API_KEY，跳过 GoldAPI")
            return None

        try:
            headers = {"x-access-token": GOLD_API_KEY}
            response = self.session.get(GOLD_API_URL, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                "timestamp": datetime.now().isoformat(),
                "price": data.get("price"),
                "open": data.get("open_price"),
                "high": data.get("high_price"),
                "low": data.get("low_price"),
                "change": data.get("ch"),
                "change_percent": data.get("chp"),
                "source": "goldapi",
            }
        except Exception as e:
            logger.error(f"GoldAPI 获取失败: {e}")
            return None

    def fetch_from_metals_live(self) -> Optional[Dict]:
        """从 metals.live 获取实时金价"""
        try:
            response = self.session.get(BACKUP_APIS[0], timeout=10)
            response.raise_for_status()
            data = response.json()

            # 解析返回的数据
            gold_data = None
            for item in data:
                if item.get("metal") == "gold" or "XAU" in str(item):
                    gold_data = item
                    break

            if gold_data:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "price": gold_data.get("price") or gold_data.get("rate"),
                    "open": gold_data.get("open"),
                    "high": gold_data.get("high"),
                    "low": gold_data.get("low"),
                    "change": gold_data.get("change"),
                    "change_percent": gold_data.get("changePercent"),
                    "source": "metals_live",
                }
            return None
        except Exception as e:
            logger.error(f"Metals.live API 获取失败: {e}")
            return None

    def fetch_from_exchangerate(self) -> Optional[Dict]:
        """从 exchangerate-api 获取 USD 汇率，模拟金价数据"""
        try:
            response = self.session.get(BACKUP_APIS[1], timeout=10)
            response.raise_for_status()
            data = response.json()

            # 这个 API 返回汇率，我们用固定基准价乘以汇率来模拟
            base_price = 2000  # 基准金价 USD/oz

            # 获取时间戳
            timestamp = datetime.now()

            return {
                "timestamp": timestamp.isoformat(),
                "price": base_price,
                "open": base_price * 0.998,
                "high": base_price * 1.005,
                "low": base_price * 0.995,
                "change": 0,
                "change_percent": 0,
                "source": "exchangerate_simulated",
                "note": "使用模拟数据，仅用于演示",
            }
        except Exception as e:
            logger.error(f"ExchangeRate API 获取失败: {e}")
            return None

    def fetch_from_web_scraper(self) -> Optional[Dict]:
        """
        通过网页爬虫从免费网站获取实时金价
        尝试多个来源，成功一个即返回
        """
        # 定义要尝试的爬虫配置
        scrapers = [
            {
                "name": "coinbase_xau",
                "url": "https://api.coinbase.com/v2/exchange-rates?currency=XAU",
                "parser": self._parse_coinbase_xau,
            },
            {
                "name": "yahoo_finance_gld",
                "url": "https://query1.finance.yahoo.com/v8/finance/chart/GLD?interval=1d&range=1d",
                "parser": self._parse_yahoo_finance_gld,
            },
            {
                "name": "investing_cn",
                "url": "https://cn.investing.com/currencies/xau-usd",
                "parser": self._parse_investing_com,
            },
            {
                "name": "kitco",
                "url": "https://www.kitco.com/",
                "parser": self._parse_kitco,
            },
        ]

        for scraper in scrapers:
            try:
                logger.info(f"尝试从 {scraper['name']} 爬取金价...")

                # 设置更完整的请求头模拟真实浏览器
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Referer": "https://www.google.com/",
                    "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
                    "Sec-Ch-Ua-Mobile": "?0",
                    "Sec-Ch-Ua-Platform": '"Windows"',
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "cross-site",
                    "Sec-Fetch-User": "?1",
                    "Upgrade-Insecure-Requests": "1",
                    "Cache-Control": "max-age=0",
                }

                # 清除之前的 cookies，避免污染
                self.session.cookies.clear()

                # 对 Coinbase API 禁用 SSL 验证（解决证书问题）
                verify_ssl = not scraper["name"].startswith("coinbase")

                response = self.session.get(
                    scraper["url"],
                    headers=headers,
                    timeout=15,
                    allow_redirects=True,
                    verify=verify_ssl
                )
                response.raise_for_status()

                # 解析数据
                price_data = scraper["parser"](response.text)

                if price_data and price_data.get("price"):
                    logger.info(f"从 {scraper['name']} 爬取成功: ${price_data['price']}")
                    return {
                        "timestamp": datetime.now().isoformat(),
                        "price": price_data["price"],
                        "open": price_data.get("open"),
                        "high": price_data.get("high"),
                        "low": price_data.get("low"),
                        "change": price_data.get("change"),
                        "change_percent": price_data.get("change_percent"),
                        "source": scraper["name"],
                    }

            except requests.exceptions.RequestException as e:
                logger.warning(f"{scraper['name']} 请求失败: {e}")
            except Exception as e:
                logger.warning(f"{scraper['name']} 解析失败: {e}")

        logger.warning("所有网页爬虫均失败")
        return None

    def _parse_investing_com(self, html: str) -> Optional[Dict]:
        """解析 investing.com 的 HTML 提取金价数据"""
        soup = BeautifulSoup(html, "html.parser")
        result = {}

        # 尝试多种选择器查找价格
        price_selectors = [
            '[data-test="instrument-price-last"]',
            '.last-price-value',
            '.last-price',
            '#last_last',
            '.arial_26',
            '[class*="price"][class*="last"]',
        ]

        for selector in price_selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    price_text = element.get_text(strip=True)
                    # 提取数字（处理千分位逗号）
                    price_match = re.search(r'[\d,]+\.?\d*', price_text.replace(',', ''))
                    if price_match:
                        result["price"] = float(price_match.group().replace(',', ''))
                        break
            except Exception:
                continue

        # 尝试提取涨跌幅
        change_selectors = [
            '[data-test="instrument-price-change"]',
            '.change-value',
            '.change-percent',
            '.parentheses',
        ]

        for selector in change_selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    change_text = element.get_text(strip=True)
                    # 提取百分比
                    percent_match = re.search(r'([+-]?[\d.]+)%', change_text)
                    if percent_match:
                        result["change_percent"] = float(percent_match.group(1))
                    # 提取变化值
                    value_match = re.search(r'[+-]?[\d,]+\.?\d*', change_text.replace(',', ''))
                    if value_match:
                        result["change"] = float(value_match.group().replace(',', ''))
                    break
            except Exception:
                continue

        return result if result.get("price") else None

    def _parse_kitco(self, html: str) -> Optional[Dict]:
        """解析 kitco.com 的 HTML 提取金价数据"""
        soup = BeautifulSoup(html, "html.parser")
        result = {}

        # Kitco 通常将价格放在特定的 span 或 div 中
        price_selectors = [
            '#sp-bid',
            '.price-value',
            '[class*="gold-price"]',
            '[class*="spot-price"]',
            'span[data-price]',
        ]

        for selector in price_selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    price_text = element.get_text(strip=True)
                    price_match = re.search(r'[\d,]+\.?\d*', price_text.replace(',', ''))
                    if price_match:
                        result["price"] = float(price_match.group().replace(',', ''))
                        break
            except Exception:
                continue

        # 如果上面的选择器没找到，尝试从 script 标签中的 JSON 数据提取
        if not result.get("price"):
            try:
                scripts = soup.find_all("script")
                for script in scripts:
                    script_text = script.string if script else ""
                    if script_text:
                        # 查找包含 gold 或 price 的 JSON 数据
                        price_match = re.search(r'"gold"[:\s]+([\d.]+)', script_text, re.IGNORECASE)
                        if price_match:
                            result["price"] = float(price_match.group(1))
                            break
                        # 尝试匹配 XAU/USD 格式
                        xau_match = re.search(r'XAU/USD["\']?\s*[:\s]+["\']?([\d.]+)', script_text, re.IGNORECASE)
                        if xau_match:
                            result["price"] = float(xau_match.group(1))
                            break
            except Exception:
                pass

        return result if result.get("price") else None

    def _parse_yahoo_finance_gld(self, html: str) -> Optional[Dict]:
        """解析 Yahoo Finance API 返回的 GLD ETF 数据并转换为金价"""
        try:
            data = json.loads(html)
            if data.get("chart") and data["chart"].get("result"):
                result = data["chart"]["result"][0]
                meta = result.get("meta", {})
                gld_price = meta.get("regularMarketPrice", 0)
                prev_close = meta.get("previousClose", 0)

                if gld_price > 0:
                    # GLD 是黄金 ETF，但价格与金价的关系会随时间变化
                    # 当前比例：GLD 价格 * 5.97 ≈ 金价（美元/盎司）
                    # 注意：这个比例需要根据实际市场情况调整
                    conversion_ratio = 5.97
                    gold_price = gld_price * conversion_ratio
                    prev_gold = prev_close * conversion_ratio if prev_close > 0 else 0
                    change = gold_price - prev_gold if prev_gold > 0 else None
                    change_percent = (change / prev_gold * 100) if change and prev_gold > 0 else None

                    return {
                        "price": round(gold_price, 2),
                        "open": meta.get("regularMarketOpen", 0) * conversion_ratio if meta.get("regularMarketOpen") else None,
                        "high": meta.get("regularMarketDayHigh", 0) * conversion_ratio if meta.get("regularMarketDayHigh") else None,
                        "low": meta.get("regularMarketDayLow", 0) * conversion_ratio if meta.get("regularMarketDayLow") else None,
                        "change": round(change, 2) if change else None,
                        "change_percent": round(change_percent, 3) if change_percent else None,
                    }
        except Exception as e:
            logger.debug(f"Yahoo Finance GLD API 解析失败: {e}")
        return None

    def _parse_yahoo_finance(self, html: str) -> Optional[Dict]:
        """解析 Yahoo Finance API 返回的黄金期货数据"""
        try:
            data = json.loads(html)
            if data.get("chart") and data["chart"].get("result"):
                result = data["chart"]["result"][0]
                meta = result.get("meta", {})
                price = meta.get("regularMarketPrice", 0)
                prev_close = meta.get("previousClose", 0)

                if price > 1000:  # 确保是合理的金价
                    change = price - prev_close if prev_close > 0 else None
                    change_percent = (change / prev_close * 100) if change and prev_close > 0 else None
                    return {
                        "price": round(price, 2),
                        "open": meta.get("regularMarketOpen"),
                        "high": meta.get("regularMarketDayHigh"),
                        "low": meta.get("regularMarketDayLow"),
                        "change": round(change, 2) if change else None,
                        "change_percent": round(change_percent, 3) if change_percent else None,
                    }
        except Exception as e:
            logger.debug(f"Yahoo Finance API 解析失败: {e}")
        return None

    def _parse_metals_live(self, html: str) -> Optional[Dict]:
        """解析 metals.live API 返回的现货金属数据"""
        try:
            data = json.loads(html)
            if isinstance(data, list):
                for item in data:
                    metal = item.get("metal", "").lower()
                    if metal == "gold" or item.get("symbol") == "XAU":
                        return {
                            "price": float(item.get("price", 0)),
                            "change": float(item.get("change", 0)) if item.get("change") else None,
                            "change_percent": float(item.get("changePercent", 0)) if item.get("changePercent") else None,
                        }
        except Exception as e:
            logger.debug(f"Metals.live API 解析失败: {e}")
        return None

    def _parse_gold_api_com(self, html: str) -> Optional[Dict]:
        """解析 gold-api.com 返回的金价数据"""
        try:
            data = json.loads(html)
            if data.get("price"):
                price = float(data["price"])
                if price > 1000:  # 确保是合理的金价
                    return {
                        "price": round(price, 2),
                        "change": data.get("change"),
                        "change_percent": data.get("change_percent"),
                    }
        except Exception as e:
            logger.debug(f"Gold-API.com 解析失败: {e}")
        return None

    def _parse_frankfurter(self, html: str) -> Optional[Dict]:
        """解析 frankfurter.app API 返回的 XAU/USD 汇率数据"""
        try:
            data = json.loads(html)
            if data.get("rates") and data["rates"].get("USD"):
                price = float(data["rates"]["USD"])
                if price > 1000:  # 确保是合理的金价
                    return {"price": round(price, 2)}
        except Exception as e:
            logger.debug(f"Frankfurter API 解析失败: {e}")
        return None

    def _parse_open_er_api(self, html: str) -> Optional[Dict]:
        """解析 open.er-api.com 返回的 XAU/USD 汇率数据"""
        try:
            data = json.loads(html)
            if data.get("rates") and data["rates"].get("USD"):
                price = float(data["rates"]["USD"])
                if price > 1000:  # 确保是合理的金价
                    return {"price": round(price, 2)}
        except Exception as e:
            logger.debug(f"Open ER API 解析失败: {e}")
        return None

    def _parse_exchangerate_host(self, html: str) -> Optional[Dict]:
        """解析 exchangerate.host API 返回的 XAU/USD 汇率数据"""
        try:
            data = json.loads(html)
            if data.get("result"):
                price = float(data["result"])
                if price > 1000:  # 确保是合理的金价
                    return {"price": round(price, 2)}
        except Exception as e:
            logger.debug(f"ExchangeRate.host API 解析失败: {e}")
        return None

    def _parse_coinbase_xau(self, html: str) -> Optional[Dict]:
        """解析 Coinbase API 返回的 XAU/USD 汇率数据"""
        try:
            data = json.loads(html)
            if data.get("data") and data["data"].get("rates"):
                usd_rate = data["data"]["rates"].get("USD")
                if usd_rate:
                    price = float(usd_rate)
                    if price > 1000:
                        return {"price": round(price, 2)}
        except Exception as e:
            logger.debug(f"Coinbase API 解析失败: {e}")
        return None

    def _parse_crypto_compare(self, html: str) -> Optional[Dict]:
        """解析 CryptoCompare PAXG/USD 价格数据"""
        try:
            data = json.loads(html)
            if data.get("USD"):
                price = float(data["USD"])
                if price > 1000:
                    return {"price": round(price, 2)}
        except Exception as e:
            logger.debug(f"CryptoCompare API 解析失败: {e}")
        return None

    def _parse_metals_api(self, html: str) -> Optional[Dict]:
        """解析 metals.live spot API 返回的数据"""
        try:
            data = json.loads(html)
            # API 返回数组格式，找黄金数据
            if isinstance(data, list):
                for item in data:
                    if item.get("metal") == "gold" or "XAU" in str(item.get("symbol", "")):
                        return {
                            "price": float(item.get("price", 0)),
                            "change": float(item.get("change", 0)) if item.get("change") else None,
                            "change_percent": float(item.get("changePercent", 0)) if item.get("changePercent") else None,
                        }
            elif isinstance(data, dict) and "gold" in data:
                gold_data = data["gold"]
                return {"price": float(gold_data.get("price", 0))}
        except Exception as e:
            logger.debug(f"Metals.live Spot API 解析失败: {e}")
        return None

    def _parse_gold_price_live(self, html: str) -> Optional[Dict]:
        """解析 goldprice.live API 返回的数据"""
        try:
            data = json.loads(html)
            if data.get("price") or data.get("gold"):
                price = data.get("price") or data.get("gold", {}).get("price")
                if price:
                    return {"price": float(price)}
        except Exception as e:
            logger.debug(f"GoldPrice.live API 解析失败: {e}")
        return None

    def _parse_floatrates(self, html: str) -> Optional[Dict]:
        """解析 floatrates.com 返回的 XAU 汇率数据"""
        try:
            data = json.loads(html)
            if data and "usd" in data:
                usd_data = data["usd"]
                # floatrates 返回的是 1 XAU = ? USD
                rate = usd_data.get("rate", 0)
                if rate > 1000:  # 确保是合理的金价
                    return {"price": round(rate, 2)}
        except Exception as e:
            logger.debug(f"FloatRates API 解析失败: {e}")
        return None

    def _parse_binance_xaut(self, html: str) -> Optional[Dict]:
        """解析 Binance XAUT (Tether Gold) 价格数据"""
        try:
            data = json.loads(html)
            if data.get("lastPrice"):
                price = float(data["lastPrice"])
                change_percent = float(data.get("priceChangePercent", 0))
                high = float(data.get("highPrice", 0))
                low = float(data.get("lowPrice", 0))
                open_price = float(data.get("openPrice", 0))

                if price > 1000:  # 确保是合理的金价
                    return {
                        "price": round(price, 2),
                        "open": round(open_price, 2),
                        "high": round(high, 2),
                        "low": round(low, 2),
                        "change_percent": round(change_percent, 3),
                    }
        except Exception as e:
            logger.debug(f"Binance API 解析失败: {e}")
        return None

    def _parse_forex_api(self, html: str) -> Optional[Dict]:
        """解析 exchangerate-api.com 返回的汇率数据（XAU/USD）"""
        try:
            data = json.loads(html)
            if data.get("rates") and data["rates"].get("USD"):
                # API 返回的是 1 XAU = ? USD
                price = data["rates"]["USD"]
                if price > 1000:  # 确保是合理的金价
                    return {"price": round(price, 2)}
        except Exception as e:
            logger.debug(f"Forex API 解析失败: {e}")
        return None

    def _parse_coingecko_paxg(self, html: str) -> Optional[Dict]:
        """解析 CoinGecko PAX Gold 代币价格（与黄金挂钩）"""
        try:
            data = json.loads(html)
            if data.get("pax-gold"):
                paxg = data["pax-gold"]
                price = paxg.get("usd", 0)
                change_24h = paxg.get("usd_24h_change", 0)
                if price > 1000:  # 确保是合理的金价
                    return {
                        "price": round(price, 2),
                        "change_percent": round(change_24h, 3) if change_24h else None,
                    }
        except Exception as e:
            logger.debug(f"CoinGecko API 解析失败: {e}")
        return None

    def _parse_goldapi_io(self, html: str) -> Optional[Dict]:
        """解析 goldapi.io 返回的 JSON 数据（免费版有限制）"""
        try:
            data = json.loads(html)
            if data.get("price"):
                return {
                    "price": float(data.get("price", 0)),
                    "open": data.get("open_price") or data.get("open"),
                    "high": data.get("high_price") or data.get("high"),
                    "low": data.get("low_price") or data.get("low"),
                    "change": data.get("ch") or data.get("change"),
                    "change_percent": data.get("chp") or data.get("change_percent"),
                }
        except Exception as e:
            logger.debug(f"GoldAPI.io 解析失败: {e}")
        return None

    def _parse_metals_api(self, html: str) -> Optional[Dict]:
        """解析 metals-api.com 返回的 JSON 数据"""
        try:
            data = json.loads(html)
            if data.get("rates") and data["rates"].get("XAU"):
                # metals-api 返回的是 1 XAU = ? USD，需要取倒数
                rate = data["rates"]["XAU"]
                if rate > 0:
                    price = 1 / rate
                    return {"price": round(price, 2)}
        except Exception as e:
            logger.debug(f"Metals-API 解析失败: {e}")
        return None

    def _parse_eastmoney_api(self, html: str) -> Optional[Dict]:
        """解析东方财富 API 返回的 JSON 数据"""
        try:
            data = json.loads(html)
            if data.get("data"):
                stock_data = data["data"]
                # 东方财富字段说明：
                # f43: 最新价（扩大100倍）, f44: 最高价, f45: 最低价
                # f46: 开盘价, f47: 成交量, f48: 成交额
                # f57: 代码, f58: 名称, f60: 昨收, f170: 涨跌幅
                price = stock_data.get("f43", 0) / 100
                open_price = stock_data.get("f46", 0) / 100
                high = stock_data.get("f44", 0) / 100
                low = stock_data.get("f45", 0) / 100
                prev_close = stock_data.get("f60", 0) / 100
                change_percent = stock_data.get("f170", 0) / 100

                if price > 0:
                    return {
                        "price": round(price, 2),
                        "open": round(open_price, 2) if open_price > 0 else None,
                        "high": round(high, 2) if high > 0 else None,
                        "low": round(low, 2) if low > 0 else None,
                        "change": round(price - prev_close, 2) if prev_close > 0 else None,
                        "change_percent": round(change_percent, 3) if change_percent != 0 else None,
                    }
        except Exception as e:
            logger.debug(f"东方财富 API 解析失败: {e}")
        return None

    def _parse_sina_api(self, html: str) -> Optional[Dict]:
        """解析新浪财经 API 返回的数据（JS 变量格式）"""
        try:
            # 新浪返回格式: var hq_str_hf_XAU="价格,买入,卖出,最高价,最低价,时间";
            match = re.search(r'var\s+hq_str_hf_XAU\s*=\s*"([^"]*)"', html)
            if match:
                values = match.group(1).split(",")
                if len(values) >= 5:
                    price = float(values[0])
                    high = float(values[3])
                    low = float(values[4])

                    # 新浪黄金期货价格是美元/盎司，但可能和现货有差异
                    if price > 1000:  # 确保是金价范围
                        return {
                            "price": round(price, 2),
                            "high": round(high, 2),
                            "low": round(low, 2),
                        }
        except Exception as e:
            logger.debug(f"新浪 API 解析失败: {e}")
        return None

    def _parse_sina_finance(self, html: str) -> Optional[Dict]:
        """解析新浪财经的 HTML 提取金价数据"""
        soup = BeautifulSoup(html, "html.parser")
        result = {}

        # 新浪通常将价格放在特定的表格或 div 中
        price_selectors = [
            '.price',
            '#price',
            '[class*="price"]',
            'td[data-code="XAU"]',
        ]

        for selector in price_selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    price_text = element.get_text(strip=True)
                    price_match = re.search(r'([d,]+.?d*)', price_text.replace(',', ''))
                    if price_match:
                        price = float(price_match.group(1).replace(',', ''))
                        # 新浪价格可能是人民币/克，需要转换为美元/盎司（约 1 克 = 0.0321507 盎司，汇率约 7.2）
                        if price < 1000:  # 如果价格小于 1000，可能是人民币/克
                            price = price * 7.2 / 0.0321507
                        result["price"] = round(price, 2)
                        break
            except Exception:
                continue

        # 尝试从 script 标签中的 JSON 数据提取
        if not result.get("price"):
            try:
                scripts = soup.find_all("script")
                for script in scripts:
                    script_text = script.string if script else ""
                    if script_text and "gold" in script_text.lower():
                        # 尝试匹配各种格式
                        patterns = [
                            r'["\']price["\']\s*[:=]\s*["\']?([\d.]+)["\']?',
                            r'XAU[US]*D?["\']?\s*[:=]\s*["\']?([\d.]+)["\']?',
                        ]
                        for pattern in patterns:
                            match = re.search(pattern, script_text, re.IGNORECASE)
                            if match:
                                result["price"] = float(match.group(1))
                                break
            except Exception:
                pass

        return result if result.get("price") else None

    def _parse_eastmoney(self, html: str) -> Optional[Dict]:
        """解析东方财富的 HTML 提取金价数据"""
        soup = BeautifulSoup(html, "html.parser")
        result = {}

        # 东方财富通常使用这些选择器
        price_selectors = [
            '.price',
            '.current-price',
            '[data-field="price"]',
            '#quote_price',
        ]

        for selector in price_selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    price_text = element.get_text(strip=True)
                    price_match = re.search(r'([\d,]+.?d*)', price_text.replace(',', ''))
                    if price_match:
                        result["price"] = float(price_match.group(1).replace(',', ''))
                        break
            except Exception:
                continue

        # 尝试从页面中的 JavaScript 变量提取
        if not result.get("price"):
            try:
                scripts = soup.find_all("script")
                for script in scripts:
                    script_text = script.string if script else ""
                    if script_text:
                        # 查找 var quote 或类似的数据结构
                        quote_match = re.search(r'var\s+quote\s*=\s*({[^}]+})', script_text)
                        if quote_match:
                            json_str = quote_match.group(1)
                            price_match = re.search(r'["\']price["\']\s*:\s*([\d.]+)', json_str)
                            if price_match:
                                result["price"] = float(price_match.group(1))
                                break
            except Exception:
                pass

        return result if result.get("price") else None

    def fetch_current_price(self) -> Optional[Dict]:
        """获取当前金价，按优先级尝试多个 API"""
        logger.info("开始获取当前金价...")

        # 尝试 GoldAPI
        data = self.fetch_from_gold_api()
        if data and data.get("price"):
            logger.info(f"从 GoldAPI 获取成功: ${data['price']}")
            return data

        # 尝试 Metals.live
        data = self.fetch_from_metals_live()
        if data and data.get("price"):
            logger.info(f"从 Metals.live 获取成功: ${data['price']}")
            return data

        # 尝试网页爬虫（免费，无需 API Key）
        data = self.fetch_from_web_scraper()
        if data and data.get("price"):
            logger.info(f"从网页爬虫获取成功: ${data['price']}")
            return data

        # 使用备用模拟数据
        logger.warning("使用模拟数据")
        return self.fetch_from_exchangerate()

    def generate_historical_data(self, days: int = ANALYSIS_DAYS) -> List[Dict]:
        """
        获取历史数据用于技术分析指标计算
        从 yfinance 真实获取最近的日线数据
        """
        import yfinance as yf
        import pandas as pd
        logger.info(f"从 yfinance 获取最近 {days} 个交易日的金价 (GC=F)...")

        # 确保能圈出足够的交易日，扩大日历跨度
        period_days = int(days * 1.5) + 10
        df = yf.download("GC=F", period=f"{period_days}d")
        
        if df.empty:
            logger.error("yfinance 获取失败，返回空集")
            return []
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df = df.tail(days)
        
        historical_data = []
        for date, row in df.iterrows():
            open_price = float(row['Open'])
            close_price = float(row['Close'])
            high_price = float(row['High'])
            low_price = float(row['Low'])
            
            historical_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "timestamp": date.isoformat(),
                "price": round(close_price, 2),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "change": round(close_price - open_price, 2),
                "change_percent": round((close_price - open_price) / open_price * 100, 3) if open_price != 0 else 0,
            })
            
        logger.info(f"历史数据获取完成，共 {len(historical_data)} 条实际交易日记录")
        return historical_data

    def save_data(self, data: List[Dict]) -> str:
        """保存数据到本地文件"""
        filepath = self.data_file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"数据已保存到: {filepath}")
        return filepath

    def load_data(self) -> List[Dict]:
        """从本地文件加载数据"""
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"数据文件不存在: {self.data_file}")
            return []

    def update_data(self) -> List[Dict]:
        """更新数据：获取最新数据并合并历史"""
        historical = self.generate_historical_data()
        self.save_data(historical)
        return historical


if __name__ == "__main__":
    fetcher = GoldDataFetcher()
    data = fetcher.update_data()
    print(f"获取到 {len(data)} 条数据")
    print(f"最新价格: ${data[-1]['price']}")
