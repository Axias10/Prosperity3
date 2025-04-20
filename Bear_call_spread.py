from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    KELP = "KELP"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"  # Added JAMS product
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 0.5,
        "clear_width": 0.5,
        "disregard_edge": 0.25,
        "join_edge": 3,
        "default_edge": 1,
        "soft_position_limit": 25,
    },
    Product.SQUID_INK: {
        "take_width": 2,
        "clear_width": 3,
        "prevent_adverse": True,
        "adverse_volume": 8,
        "reversion_beta": -0.229,
        "disregard_edge": 2,
        "join_edge": 1,
        "default_edge": 3,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 20,
        "reversion_beta": -0.229,
        "disregard_edge": 2,
        "join_edge": 1,
        "default_edge": 4,
    },
    Product.CROISSANTS: {
        "take_width": 1.5,
        "clear_width": 1.0,
        "mean_reversion_strength": -0.3,
        "history_window": 20,
        "sinc_peak_threshold": 0.5,
        "sinc_center": 10000,
        "sinc_width": 500,
        "position_limit": 40,
        "buy_threshold": 0.4,
        "sell_threshold": 0.1,
        "max_spread": 10,
    },
    Product.JAMS: {  # Added JAMS with same parameters as CROISSANTS
        "take_width": 1.5,
        "clear_width": 1.0,
        "mean_reversion_strength": -0.3,
        "history_window": 20,
        "sinc_peak_threshold": 0.5,
        "sinc_center": 10000,
        "sinc_width": 500,
        "position_limit": 40,
        "buy_threshold": 0.4,
        "sell_threshold": 0.1,
        "max_spread": 10,
    },
    Product.VOLCANIC_ROCK: {
        "position_limit": 50  # Position limit for the underlying
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "strike": 10250,
        "position_limit": 10,
        "max_hold_time": 5  # Maximum holding time for the spread position
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "strike": 10500,
        "position_limit": 10,
        "max_hold_time": 5
    }
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK: 50,
            Product.KELP: 50,
            Product.CROISSANTS: 40,
            Product.JAMS: 40,  # Added position limit for JAMS
            Product.VOLCANIC_ROCK: 50,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 10,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 10
        }

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def SQUID_INK_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("SQUID_INK_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["SQUID_INK_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("SQUID_INK_last_price", None) != None:
                last_price = traderObject["SQUID_INK_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.SQUID_INK]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["SQUID_INK_last_price"] = mmmid_price
            return fair
        return None
    
    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("KELP_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("KELP_last_price", None) != None:
                last_price = traderObject["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["KELP_last_price"] = mmmid_price
            return fair
        return None

    def sinc_fair_value(self, product: str, order_depth: OrderDepth, traderObject) -> float:
        """
        Calculate fair value for products using sinc function density model and negative autocorrelation
        (Used for CROISSANTS and JAMS)
        """
        price_history_key = f"{product}_price_history"
        if price_history_key not in traderObject:
            traderObject[price_history_key] = []
            
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            
            traderObject[price_history_key].append(mid_price)
            
            history_window = self.params[product]["history_window"]
            if len(traderObject[price_history_key]) > history_window:
                traderObject[price_history_key] = traderObject[price_history_key][-history_window:]
            
            if len(traderObject[price_history_key]) > 1:
                last_price = traderObject[price_history_key][-2]
                current_price = traderObject[price_history_key][-1]
                
                center = self.params[product]["sinc_center"]
                deviation = current_price - center
                
                sinc_width = self.params[product]["sinc_width"]
                if deviation == 0:
                    sinc_value = 1.0
                else:
                    x = deviation / sinc_width
                    sinc_value = np.sin(np.pi * x) / (np.pi * x) if x != 0 else 1.0
                
                traderObject[f"{product}_sinc_value"] = sinc_value
                
                mean_reversion_strength = self.params[product]["mean_reversion_strength"]
                price_change = current_price - last_price
                predicted_change = price_change * mean_reversion_strength
                fair_value = current_price + predicted_change
                
                traderObject[f"{product}_predicted_direction"] = np.sign(predicted_change)
                
                return fair_value
            
            return mid_price
        
        return None

    def sinc_zero_crossings(self, x):
        """Calculate approximate zero crossings of the sinc function"""
        return [n for n in range(1, 10)]

    def sinc_mean_reversion_strategy(
        self,
        product: str,
        order_depth: OrderDepth,
        traderObject,
        position: int
    ) -> List[Order]:
        """Implement mean reversion strategy based on sinc function for specified product"""
        orders = []
        
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return orders
            
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        current_spread = best_ask - best_bid
        
        sinc_value = traderObject.get(f"{product}_sinc_value", 0)
        predicted_direction = traderObject.get(f"{product}_predicted_direction", 0)
        
        center = self.params[product]["sinc_center"]
        current_price = traderObject[f"{product}_price_history"][-1]
        
        buy_threshold = self.params[product]["buy_threshold"]
        sell_threshold = self.params[product]["sell_threshold"]
        position_limit = self.params[product]["position_limit"]
        
        deviation = current_price - center
        sinc_width = self.params[product]["sinc_width"]
        normalized_deviation = deviation / sinc_width
        
        zero_crossings = self.sinc_zero_crossings(normalized_deviation)
        distances = [abs(normalized_deviation - zc) for zc in zero_crossings] + [abs(normalized_deviation + zc) for zc in zero_crossings]
        nearest_zero_distance = min(distances) if distances else float('inf')
        
        buy_quantity = position_limit - position
        sell_quantity = position_limit + position
        
        if abs(sinc_value) > buy_threshold and predicted_direction > 0 and position < position_limit:
            price = best_bid + 1
            if buy_quantity > 0:
                orders.append(Order(product, price, buy_quantity))
        elif abs(sinc_value) < sell_threshold or predicted_direction < 0:
            price = best_bid - 1 if position > 0 else best_ask - 1
            if sell_quantity > 0:
                orders.append(Order(product, price, -sell_quantity))
        else:
            max_spread = self.params[product]["max_spread"]
            mid_price = (best_ask + best_bid) / 2
            spread_factor = 1 - abs(sinc_value) / 2
            spread = max(1, min(max_spread, current_spread) * spread_factor)
            ask = round(mid_price + spread / 2)
            bid = round(mid_price - spread / 2)
            if buy_quantity > 0:
                orders.append(Order(product, bid, buy_quantity))
            if sell_quantity > 0:
                orders.append(Order(product, ask, -sell_quantity))
        
        return orders
    
    def bear_call_spread_strategy(
        self,
        option_low: str,
        option_high: str,
        option_low_depth: OrderDepth,
        option_high_depth: OrderDepth,
        traderObject: dict,
        positions: dict,
        timestamp: int
    ) -> dict:
        """
        Implement a Bear Call Spread strategy:
        - Sell 1 call at a lower strike (option_low).
        - Buy 1 call at a higher strike (option_high).
        """
        orders = {
            option_low: [],
            option_high: []
        }

        # Skip if not enough orders in the books
        if (len(option_low_depth.sell_orders) == 0 or len(option_low_depth.buy_orders) == 0 or
            len(option_high_depth.sell_orders) == 0 or len(option_high_depth.buy_orders) == 0):
            return orders

        # Get current positions
        position_low = positions.get(option_low, 0)
        position_high = positions.get(option_high, 0)

        # Position tracking keys
        position_key = f"{option_low}_bear_call_spread_position"
        entry_time_key = f"{option_low}_bear_call_spread_entry_time"

        # Check if we already have a Bear Call Spread position
        spread_position = traderObject.get(position_key, 0)

        # If we have an existing position, manage it
        if spread_position != 0:
            entry_time = traderObject.get(entry_time_key)

            # Check time-based exit
            hold_time = timestamp - entry_time
            max_hold_time = self.params[option_low]["max_hold_time"]
            if hold_time >= max_hold_time:
                # Close the position: reverse the original trades
                if position_low < 0:
                    best_ask_low = min(option_low_depth.sell_orders.keys())
                    orders[option_low].append(Order(option_low, best_ask_low, -position_low))
                if position_high > 0:
                    best_bid_high = max(option_high_depth.buy_orders.keys())
                    orders[option_high].append(Order(option_high, best_bid_high, -position_high))
                traderObject[position_key] = 0
                traderObject.pop(entry_time_key, None)
            return orders

        # If no position, check if we can enter a new spread
        position_limit_low = self.params[option_low]["position_limit"]
        position_limit_high = self.params[option_high]["position_limit"]

        if (position_low > -position_limit_low and
            position_high < position_limit_high):

            # Get best prices
            best_bid_low = max(option_low_depth.buy_orders.keys())
            best_ask_high = min(option_high_depth.sell_orders.keys())

            # Calculate credit
            credit = best_bid_low - best_ask_high
            if credit > 0:  # Only enter if we receive a net credit
                # Sell 1 call at lower strike (10250)
                orders[option_low].append(Order(option_low, best_bid_low, -1))
                # Buy 1 call at higher strike (10500)
                orders[option_high].append(Order(option_high, best_ask_high, 1))

                # Update position tracking
                traderObject[position_key] = 1  # Indicates we have a spread position
                traderObject[entry_time_key] = timestamp

        return orders

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            amethyst_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            amethyst_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    amethyst_position,
                )
            )
            amethyst_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    amethyst_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            amethyst_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                amethyst_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                amethyst_take_orders + amethyst_clear_orders + amethyst_make_orders
            )

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            SQUID_INK_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
            SQUID_INK_fair_value = self.SQUID_INK_fair_value(
                state.order_depths[Product.SQUID_INK], traderObject
            )
            SQUID_INK_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    SQUID_INK_fair_value,
                    self.params[Product.SQUID_INK]["take_width"],
                    SQUID_INK_position,
                    self.params[Product.SQUID_INK]["prevent_adverse"],
                    self.params[Product.SQUID_INK]["adverse_volume"],
                )
            )
            SQUID_INK_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    SQUID_INK_fair_value,
                    self.params[Product.SQUID_INK]["clear_width"],
                    SQUID_INK_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            SQUID_INK_make_orders, _, _ = self.make_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                SQUID_INK_fair_value,
                SQUID_INK_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID_INK]["disregard_edge"],
                self.params[Product.SQUID_INK]["join_edge"],
                self.params[Product.SQUID_INK]["default_edge"],
            )
            result[Product.SQUID_INK] = (
                SQUID_INK_take_orders + SQUID_INK_clear_orders + SQUID_INK_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            KELP_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            KELP_fair_value = self.KELP_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            KELP_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["take_width"],
                    KELP_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            KELP_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    KELP_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            KELP_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                KELP_take_orders + KELP_clear_orders + KELP_make_orders
            )

        # Handle CROISSANTS with sinc-based mean reversion strategy
        if Product.CROISSANTS in self.params and Product.CROISSANTS in state.order_depths:
            croissants_position = (
                state.position[Product.CROISSANTS]
                if Product.CROISSANTS in state.position
                else 0
            )
            croissants_fair_value = self.sinc_fair_value(
                Product.CROISSANTS, state.order_depths[Product.CROISSANTS], traderObject
            )
            croissants_orders = self.sinc_mean_reversion_strategy(
                Product.CROISSANTS,
                state.order_depths[Product.CROISSANTS],
                traderObject,
                croissants_position,
            )
            result[Product.CROISSANTS] = croissants_orders

        # Handle JAMS with the same sinc-based mean reversion strategy
        if Product.JAMS in self.params and Product.JAMS in state.order_depths:
            jams_position = (
                state.position[Product.JAMS]
                if Product.JAMS in state.position
                else 0
            )
            jams_fair_value = self.sinc_fair_value(
                Product.JAMS, state.order_depths[Product.JAMS], traderObject
            )
            jams_orders = self.sinc_mean_reversion_strategy(
                Product.JAMS,
                state.order_depths[Product.JAMS],
                traderObject,
                jams_position,
            )
            result[Product.JAMS] = jams_orders
            
                # Handle VOLCANIC_ROCK and Bear Call Spread strategy
        if (Product.VOLCANIC_ROCK in state.order_depths and
            Product.VOLCANIC_ROCK_VOUCHER_10250 in state.order_depths and
            Product.VOLCANIC_ROCK_VOUCHER_10500 in state.order_depths):
            # Get positions
            positions = {
                Product.VOLCANIC_ROCK: state.position.get(Product.VOLCANIC_ROCK, 0),
                Product.VOLCANIC_ROCK_VOUCHER_10250: state.position.get(Product.VOLCANIC_ROCK_VOUCHER_10250, 0),
                Product.VOLCANIC_ROCK_VOUCHER_10500: state.position.get(Product.VOLCANIC_ROCK_VOUCHER_10500, 0)
            }
            # Execute Bear Call Spread strategy
            spread_orders = self.bear_call_spread_strategy(
                Product.VOLCANIC_ROCK_VOUCHER_10250,
                Product.VOLCANIC_ROCK_VOUCHER_10500,
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10250],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10500],
                traderObject,
                positions,
                state.timestamp
            )
            # Add orders to the result
            result[Product.VOLCANIC_ROCK_VOUCHER_10250] = spread_orders[Product.VOLCANIC_ROCK_VOUCHER_10250]
            result[Product.VOLCANIC_ROCK_VOUCHER_10500] = spread_orders[Product.VOLCANIC_ROCK_VOUCHER_10500]

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData