# -*- coding: utf-8 -*-
"""round_5_trader_specific.py

Optimized for IMC Prosperity Round 5 to trade specific traders on their best/worst assets.
"""

from datamodel import Order, TradingState, OrderDepth
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)

class Trader:
    def __init__(self):
        self.position_limits = {
            "JAMS": 100,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "DJEMBES": 60,
            "CROISSANTS": 250,
            "VOLCANIC_ROCK": 400,
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
        }
        self.target_products = {
            "Paris": {"JAMS": 5},
            "Camilla": {"VOLCANIC_ROCK_VOUCHER_10000": 5, "VOLCANIC_ROCK_VOUCHER_10250": 5},
            "Caesar": {"DJEMBES": 10, "CROISSANTS": 5, "VOLCANIC_ROCK": 10},
            "Charlie": {"RAINFOREST_RESIN": 5, "KELP": 5, "SQUID_INK": 5},
            "Pablo": {"VOLCANIC_ROCK_VOUCHER_9750": 5},
        }
        self.trade_memory = {}  # product -> list of (trader, quantity, timestamp, is_buy)
        self.entry_prices = {}  # (product, timestamp) -> entry price
        self.logger = logging.getLogger(__name__)

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        positions = state.position
        current_timestamp = state.timestamp

        for product, order_depth in state.order_depths.items():
            # Skip if product isn't targeted
            if not any(product in trader_products for trader_products in self.target_products.values()):
                continue

            orders[product] = []
            pos = positions.get(product, 0)
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            limit = self.position_limits.get(product, 50)

            if not (best_bid and best_ask):
                self.logger.info(f"No valid bid/ask for {product} at {current_timestamp}")
                continue

            # Update trade memory
            for trade in state.market_trades.get(product, []):
                if trade.buyer in self.target_products:
                    self.trade_memory.setdefault(product, []).append(
                        (trade.buyer, trade.quantity, current_timestamp, True)
                    )
                if trade.seller in self.target_products:
                    self.trade_memory.setdefault(product, []).append(
                        (trade.seller, trade.quantity, current_timestamp, False)
                    )
                if len(self.trade_memory[product]) > 50:
                    self.trade_memory[product].pop(0)

            # Filter recent trades (last 1000 timestamps)
            recent_trades = [
                (t, q, ts, is_buy) for t, q, ts, is_buy in self.trade_memory.get(product, [])
                if current_timestamp - ts <= 1000
            ]

            # Process each trader
            for trader, products in self.target_products.items():
                if product not in products:
                    continue
                volume = sum(q for t, q, _, is_buy in recent_trades if t == trader and is_buy)
                sell_volume = sum(q for t, q, _, is_buy in recent_trades if t == trader and not is_buy)
                self.logger.info(
                    f"{product}: {trader} buy vol {volume}, sell vol {sell_volume}, pos {pos}"
                )

                # Paris: Contrarian Short on JAMS
                if trader == "Paris" and volume >= products[product] and best_bid:
                    qty = min(5, limit + pos)
                    if pos > -limit and qty > 0:
                        orders[product].append(Order(product, best_bid, -qty))
                        self.entry_prices[(product, current_timestamp)] = best_bid
                        self.logger.info(f"Short {product} (Paris): {qty} at {best_bid}, pos {pos}")
                    # Exit at 2% drop or 1% rise (stop-loss)
                    for (p, ts), entry in list(self.entry_prices.items()):
                        if p == product and pos < 0:
                            if best_ask and best_ask <= entry * 0.98:  # 2% drop
                                qty = min(-pos, limit + pos)
                                if qty > 0:
                                    orders[product].append(Order(product, best_ask, qty))
                                    self.logger.info(f"Exit short {product} (Paris): {qty} at {best_ask}")
                                    del self.entry_prices[(p, ts)]
                            elif best_ask and best_ask >= entry * 1.01:  # 1% rise
                                qty = min(-pos, limit + pos)
                                if qty > 0:
                                    orders[product].append(Order(product, best_ask, qty))
                                    self.logger.info(f"Stop-loss short {product} (Paris): {qty} at {best_ask}")
                                    del self.entry_prices[(p, ts)]

                # Camilla: Mean-Reversion Long on VOUCHERS
                elif trader == "Camilla" and volume >= products[product] and best_ask:
                    qty = min(5, limit - pos)
                    if pos < limit and qty > 0:
                        orders[product].append(Order(product, best_ask, qty))
                        self.entry_prices[(product, current_timestamp)] = best_ask
                        self.logger.info(f"Long {product} (Camilla): {qty} at {best_ask}, pos {pos}")
                    # Exit at 5% rise
                    for (p, ts), entry in list(self.entry_prices.items()):
                        if p == product and pos > 0 and best_bid and best_bid >= entry * 1.05:
                            qty = min(pos, limit - pos)
                            if qty > 0:
                                orders[product].append(Order(product, best_bid, -qty))
                                self.logger.info(f"Exit long {product} (Camilla): {qty} at {best_bid}")
                                del self.entry_prices[(p, ts)]

                # Caesar: Momentum Short on DJEMBES, CROISSANTS, VOLCANIC_ROCK
                elif trader == "Caesar" and sell_volume >= products[product] and best_bid:
                    qty = min(5, limit + pos)
                    if pos > -limit and qty > 0:
                        orders[product].append(Order(product, best_bid, -qty))
                        self.entry_prices[(product, current_timestamp)] = best_bid
                        self.logger.info(f"Short {product} (Caesar): {qty} at {best_bid}, pos {pos}")
                    # Exit at 1.5% drop or 1% rise (stop-loss)
                    for (p, ts), entry in list(self.entry_prices.items()):
                        if p == product and pos < 0:
                            if best_ask and best_ask <= entry * 0.985:  # 1.5% drop
                                qty = min(-pos, limit + pos)
                                if qty > 0:
                                    orders[product].append(Order(product, best_ask, qty))
                                    self.logger.info(f"Exit short {product} (Caesar): {qty} at {best_ask}")
                                    del self.entry_prices[(p, ts)]
                            elif best_ask and best_ask >= entry * 1.01:  # 1% rise (stop-loss)
                                qty = min(-pos, limit + pos)
                                if qty > 0:
                                    orders[product].append(Order(product, best_ask, qty))
                                    self.logger.info(f"Stop-loss short {product} (Caesar): {qty} at {best_ask}")
                                    del self.entry_prices[(p, ts)]

                # Charlie: Momentum Long on RAINFOREST_RESIN, KELP, SQUID_INK
                elif trader == "Charlie" and volume >= products[product] and best_ask:
                    qty = min(5, limit - pos)
                    if pos < limit and qty > 0:
                        orders[product].append(Order(product, best_ask, qty))
                        self.entry_prices[(product, current_timestamp)] = best_ask
                        self.logger.info(f"Long {product} (Charlie): {qty} at {best_ask}, pos {pos}")
                    # Exit at 3% rise
                    for (p, ts), entry in list(self.entry_prices.items()):
                        if p == product and pos > 0 and best_bid and best_bid >= entry * 1.03:
                            qty = min(pos, limit - pos)
                            if qty > 0:
                                orders[product].append(Order(product, best_bid, -qty))
                                self.logger.info(f"Exit long {product} (Charlie): {qty} at {best_bid}")
                                del self.entry_prices[(p, ts)]

                # Pablo: Contrarian on VOLCANIC_ROCK_VOUCHER_9750
                elif trader == "Pablo":
                    if volume >= products[product] and best_bid:
                        qty = min(5, limit + pos)
                        if pos > -limit and qty > 0:
                            orders[product].append(Order(product, best_bid, -qty))
                            self.entry_prices[(product, current_timestamp)] = best_bid
                            self.logger.info(f"Short {product} (Pablo buy): {qty} at {best_bid}, pos {pos}")
                        # Exit at 2% drop
                        for (p, ts), entry in list(self.entry_prices.items()):
                            if p == product and pos < 0 and best_ask and best_ask <= entry * 0.98:
                                qty = min(-pos, limit + pos)
                                if qty > 0:
                                    orders[product].append(Order(product, best_ask, qty))
                                    self.logger.info(f"Exit short {product} (Pablo buy): {qty} at {best_ask}")
                                    del self.entry_prices[(p, ts)]
                    elif sell_volume >= products[product] and best_ask:
                        qty = min(5, limit - pos)
                        if pos < limit and qty > 0:
                            orders[product].append(Order(product, best_ask, qty))
                            self.entry_prices[(product, current_timestamp)] = best_ask
                            self.logger.info(f"Long {product} (Pablo sell): {qty} at {best_ask}, pos {pos}")
                        # Exit at 2% rise
                        for (p, ts), entry in list(self.entry_prices.items()):
                            if p == product and pos > 0 and best_bid and best_bid >= entry * 1.02:
                                qty = min(pos, limit - pos)
                                if qty > 0:
                                    orders[product].append(Order(product, best_bid, -qty))
                                    self.logger.info(f"Exit long {product} (Pablo sell): {qty} at {best_bid}")
                                    del self.entry_prices[(p, ts)]

        return orders, 0, ""