"""
Binance WebSocket streaming for real-time cryptocurrency data.

Provides real-time OHLCV data from Binance exchange via WebSocket connection.
"""

import asyncio
import json
from datetime import datetime
from typing import Callable, Optional

import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from pydantic import BaseModel, Field

from star_e.config import settings


class KlineData(BaseModel):
    """Real-time kline/candlestick data from Binance."""

    symbol: str
    interval: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime
    quote_volume: float
    trades: int
    taker_buy_base: float
    taker_buy_quote: float
    is_closed: bool


class BinanceStreamManager:
    """
    Manages WebSocket connections to Binance for real-time data streaming.

    Supports multiple symbol subscriptions and automatic reconnection.

    Attributes:
        api_key: Binance API key
        api_secret: Binance API secret
        symbols: List of trading pairs to subscribe to
        interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        symbols: list[str] = None,
        interval: str = "1m",
    ):
        self.api_key = api_key or settings.binance_api_key
        self.api_secret = api_secret or settings.binance_api_secret
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.interval = interval

        self.client: Optional[AsyncClient] = None
        self.bsm: Optional[BinanceSocketManager] = None
        self.callbacks: list[Callable[[KlineData], None]] = []
        self._running = False
        self._buffer: list[KlineData] = []
        self._max_buffer_size = 10000

    async def connect(self) -> None:
        """Establish connection to Binance WebSocket."""
        self.client = await AsyncClient.create(self.api_key, self.api_secret)
        self.bsm = BinanceSocketManager(self.client)
        self._running = True

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        if self.client:
            await self.client.close_connection()

    def add_callback(self, callback: Callable[[KlineData], None]) -> None:
        """Register callback for incoming kline data."""
        self.callbacks.append(callback)

    def _process_message(self, msg: dict) -> Optional[KlineData]:
        """Parse WebSocket message into KlineData."""
        if msg.get("e") != "kline":
            return None

        k = msg["k"]
        return KlineData(
            symbol=k["s"],
            interval=k["i"],
            open_time=datetime.fromtimestamp(k["t"] / 1000),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            close_time=datetime.fromtimestamp(k["T"] / 1000),
            quote_volume=float(k["q"]),
            trades=k["n"],
            taker_buy_base=float(k["V"]),
            taker_buy_quote=float(k["Q"]),
            is_closed=k["x"],
        )

    async def _handle_socket(self, socket) -> None:
        """Handle incoming WebSocket messages."""
        async with socket as stream:
            while self._running:
                try:
                    msg = await asyncio.wait_for(stream.recv(), timeout=30)
                    kline = self._process_message(msg)

                    if kline:
                        self._buffer.append(kline)
                        if len(self._buffer) > self._max_buffer_size:
                            self._buffer = self._buffer[-self._max_buffer_size :]

                        for callback in self.callbacks:
                            try:
                                callback(kline)
                            except Exception:
                                pass

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    if self._running:
                        await asyncio.sleep(5)
                        break

    async def start_stream(self) -> None:
        """Start streaming kline data for all subscribed symbols."""
        if not self.bsm:
            await self.connect()

        streams = [f"{s.lower()}@kline_{self.interval}" for s in self.symbols]
        socket = self.bsm.multiplex_socket(streams)

        await self._handle_socket(socket)

    async def start_single_symbol(self, symbol: str) -> None:
        """Start streaming for a single symbol."""
        if not self.bsm:
            await self.connect()

        socket = self.bsm.kline_socket(symbol, interval=self.interval)
        await self._handle_socket(socket)

    def get_buffer_df(self) -> pd.DataFrame:
        """Convert buffer to DataFrame."""
        if not self._buffer:
            return pd.DataFrame()

        data = [
            {
                "symbol": k.symbol,
                "timestamp": k.open_time,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume,
            }
            for k in self._buffer
        ]

        return pd.DataFrame(data)

    def clear_buffer(self) -> None:
        """Clear the kline buffer."""
        self._buffer = []


class BinanceHistoricalFetcher:
    """Fetch historical kline data from Binance REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self.api_key = api_key or settings.binance_api_key
        self.api_secret = api_secret or settings.binance_api_secret

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: str = None,
        end_date: str = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch historical kline data.

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            limit: Maximum number of klines

        Returns:
            DataFrame with OHLCV data
        """
        client = await AsyncClient.create(self.api_key, self.api_secret)

        try:
            klines = await client.get_historical_klines(
                symbol,
                interval,
                start_date,
                end_date,
                limit=limit,
            )

            df = pd.DataFrame(
                klines,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            df["symbol"] = symbol

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]

        finally:
            await client.close_connection()

    async def fetch_multiple_symbols(
        self,
        symbols: list[str],
        interval: str = "1d",
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Fetch historical data for multiple symbols."""
        tasks = [
            self.fetch_klines(s, interval, start_date, end_date) for s in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        dfs = [r for r in results if isinstance(r, pd.DataFrame)]
        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)


async def stream_to_storage(
    symbols: list[str],
    interval: str = "1m",
    duration_seconds: int = 3600,
    storage_callback: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Stream data for a fixed duration and return as DataFrame.

    Args:
        symbols: List of trading pairs
        interval: Kline interval
        duration_seconds: How long to stream
        storage_callback: Optional callback for each kline

    Returns:
        DataFrame with all collected klines
    """
    manager = BinanceStreamManager(symbols=symbols, interval=interval)

    if storage_callback:
        manager.add_callback(storage_callback)

    await manager.connect()

    stream_task = asyncio.create_task(manager.start_stream())

    await asyncio.sleep(duration_seconds)

    await manager.disconnect()
    stream_task.cancel()

    return manager.get_buffer_df()
