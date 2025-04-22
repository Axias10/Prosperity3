from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
from math import log, sqrt, exp
from statistics import NormalDist
import logging


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN" # --> Lucas
    MACARONS = "MAGNIFICENT_MACARONS" #augmenter le Pnl --> David
    GIFT_BASKET = "PICNIC_BASKET2" # Rajouter l'autre ETF --> David
    GIFT_BASKET2 = "PICNIC_BASKET1"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES" # --> David
    CROISSANTS = "CROISSANTS" #intégrer algo du sinus card du round3 et trader les trois assets --> Justin
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"
    SYNTHETIC2 = "SYNTHETIC2"
    SPREAD2 = "SPREAD2"
    

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 0,
    },
    Product.SPREAD: {
        "default_spread_mean": 76.03233333333333, 
        "default_spread_std": 63.994517586781434,
        "spread_std_window": 20,
        "zscore_threshold": 5,
        "target_position": 100,
    },
    Product.SPREAD2: {
        "default_spread_mean": -38.25885, 
        "default_spread_std": 120.39804377415119,
        "spread_std_window": 15,
        "zscore_threshold": 5,
        "target_position": 70,
    },    
    Product.MACARONS: {
        "make_edge": 2, #2
        "make_min_edge": 1,
        "make_probability": 0.566,
        "init_make_edge": 2,
        "min_edge": 0.5,
        "volume_avg_timestamp": 30, #30
        "volume_bar": 40,  # Adjusted for position limit 75 vs 100 for ORCHIDS
        "dec_edge_discount": 0.8,
        "step_size": 0.5
    }
}

BASKET_WEIGHTS = {
    Product.JAMS: 2,
    Product.CROISSANTS: 4,
}

BASKET_WEIGHTS2 = {
    Product.JAMS: 3,
    Product.CROISSANTS: 6,
    Product.DJEMBES: 1,
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.MACARONS: 20 #75
        }
        self.CONVERSION_LIMIT = {Product.MACARONS: 10}

        self.position_limits = {
            "JAMS": 100,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "DJEMBES": 60,
            "CROISSANTS": 250,
            "VOLCANIC_ROCK": 400,
            "SQUID_INK": 50,
        }
        self.target_products = {
            "Paris": {"JAMS": 85},
            "Caesar": {"DJEMBES": 5, "CROISSANTS": 6, "VOLCANIC_ROCK": 5},
            "Charlie": {"SQUID_INK": 1},
            "Pablo": {"VOLCANIC_ROCK_VOUCHER_9750": 150},
        }
        self.trade_memory = {}  # product -> list of (trader, quantity, timestamp, is_buy)
        self.entry_prices = {}  # (product, timestamp) -> entry price
        self.logger = logging.getLogger(__name__)

    def macarons_implied_bid_ask(self, observation: ConversionObservation) -> (float, float):
        # Implied prices based on external factors (similar to ORCHIDS)
        return (
            observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1,
            observation.askPrice + observation.importTariff + observation.transportFees
        )

    def macarons_adap_edge(
        self,
        timestamp: int,
        curr_edge: float,
        position: int,
        traderObject: dict
    ) -> float:
        if timestamp == 0:
            traderObject[Product.MACARONS]["curr_edge"] = self.params[Product.MACARONS]["init_make_edge"]
            return self.params[Product.MACARONS]["init_make_edge"]

        traderObject[Product.MACARONS]["volume_history"].append(abs(position))
        if len(traderObject[Product.MACARONS]["volume_history"]) > self.params[Product.MACARONS]["volume_avg_timestamp"]:
            traderObject[Product.MACARONS]["volume_history"].pop(0)

        if len(traderObject[Product.MACARONS]["volume_history"]) < self.params[Product.MACARONS]["volume_avg_timestamp"]:
            return curr_edge
        elif not traderObject[Product.MACARONS]["optimized"]:
            volume_avg = np.mean(traderObject[Product.MACARONS]["volume_history"])

            if volume_avg >= self.params[Product.MACARONS]["volume_bar"]:
                traderObject[Product.MACARONS]["volume_history"] = []
                traderObject[Product.MACARONS]["curr_edge"] = curr_edge + self.params[Product.MACARONS]["step_size"]
                return curr_edge + self.params[Product.MACARONS]["step_size"]

            elif self.params[Product.MACARONS]["dec_edge_discount"] * self.params[Product.MACARONS]["volume_bar"] * (curr_edge - self.params[Product.MACARONS]["step_size"]) > volume_avg * curr_edge:
                if curr_edge - self.params[Product.MACARONS]["step_size"] > self.params[Product.MACARONS]["min_edge"]:
                    traderObject[Product.MACARONS]["volume_history"] = []
                    traderObject[Product.MACARONS]["curr_edge"] = curr_edge - self.params[Product.MACARONS]["step_size"]
                    traderObject[Product.MACARONS]["optimized"] = True
                    return curr_edge - self.params[Product.MACARONS]["step_size"]
                else:
                    traderObject[Product.MACARONS]["curr_edge"] = self.params[Product.MACARONS]["min_edge"]
                    return self.params[Product.MACARONS]["min_edge"]

        traderObject[Product.MACARONS]["curr_edge"] = curr_edge
        return curr_edge

    def macarons_arb_take(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        adap_edge: float,
        position: int
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MACARONS]
        buy_order_volume = 0
        sell_order_volume = 0

        implied_bid, implied_ask = self.macarons_implied_bid_ask(observation)

        buy_quantity = position_limit - position
        sell_quantity = position_limit + position

        ask = implied_ask + adap_edge

        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6

        if aggressive_ask > implied_ask:
            ask = aggressive_ask

        edge = (ask - implied_ask) * self.params[Product.MACARONS]["make_probability"]

        for price in sorted(list(order_depth.sell_orders.keys())):
            if price > implied_bid - edge:
                break
            if price < implied_bid - edge:
                quantity = min(abs(order_depth.sell_orders[price]), buy_quantity)
                if quantity > 0:
                    orders.append(Order(Product.MACARONS, round(price), quantity))
                    buy_order_volume += quantity

        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price < implied_ask + edge:
                break
            if price > implied_ask + edge:
                quantity = min(abs(order_depth.buy_orders[price]), sell_quantity)
                if quantity > 0:
                    orders.append(Order(Product.MACARONS, round(price), -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def macarons_arb_clear(self, position: int) -> int:
        conversion_limit = self.CONVERSION_LIMIT[Product.MACARONS]
        conversions = max(min(-position, conversion_limit), -conversion_limit)
        return conversions

    def macarons_arb_make(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
        edge: float,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MACARONS]

        implied_bid, implied_ask = self.macarons_implied_bid_ask(observation)

        bid = implied_bid - edge
        ask = implied_ask + edge

        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6

        if aggressive_ask >= implied_ask + self.params[Product.MACARONS]["min_edge"]:
            ask = aggressive_ask

        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 30]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 20]

        if len(filtered_ask) > 0 and ask > filtered_ask[0]:
            if filtered_ask[0] - 1 > implied_ask:
                ask = filtered_ask[0] - 1
            else:
                ask = implied_ask + edge
        if len(filtered_bid) > 0 and bid < filtered_bid[0]:
            if filtered_bid[0] + 1 < implied_bid:
                bid = filtered_bid[0] + 1
            else:
                bid = implied_bid - edge

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.MACARONS, round(bid), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.MACARONS, round(ask), -sell_quantity))

        return orders, buy_order_volume, sell_order_volume
    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        JAMS_PER_BASKET = BASKET_WEIGHTS[Product.JAMS]
        CROISSANTS_PER_BASKET = BASKET_WEIGHTS[Product.CROISSANTS]


        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        JAMS_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        JAMS_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        CROISSANTS_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        CROISSANTS_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            JAMS_best_bid * JAMS_PER_BASKET
            + CROISSANTS_best_bid * CROISSANTS_PER_BASKET
        )
        implied_ask = (
            JAMS_best_ask * JAMS_PER_BASKET
            + CROISSANTS_best_ask * CROISSANTS_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            JAMS_bid_volume = (
                order_depths[Product.JAMS].buy_orders[JAMS_best_bid]
                // JAMS_PER_BASKET
            )
            CROISSANTS_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[CROISSANTS_best_bid]
                // CROISSANTS_PER_BASKET
            )
            implied_bid_volume = min(
                JAMS_bid_volume, CROISSANTS_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            JAMS_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[JAMS_best_ask]
                // JAMS_PER_BASKET
            )
            CROISSANTS_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[CROISSANTS_best_ask]
                // CROISSANTS_PER_BASKET
            )

            implied_ask_volume = min(
                JAMS_ask_volume, CROISSANTS_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.JAMS: [],
            Product.CROISSANTS: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                JAMS_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                CROISSANTS_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
            
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                JAMS_price = max(order_depths[Product.JAMS].buy_orders.keys())
                CROISSANTS_price = max(
                    order_depths[Product.CROISSANTS].buy_orders.keys()
                )
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            JAMS_order = Order(
                Product.JAMS,
                JAMS_price,
                quantity * BASKET_WEIGHTS[Product.JAMS],
            )
            CROISSANTS_order = Order(
                Product.CROISSANTS,
                CROISSANTS_price,
                quantity * BASKET_WEIGHTS[Product.CROISSANTS],
            )

            # Add the component orders to the respective lists
            component_orders[Product.JAMS].append(JAMS_order)
            component_orders[Product.CROISSANTS].append(CROISSANTS_order)

        return component_orders

    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.GIFT_BASKET]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.GIFT_BASKET, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.GIFT_BASKET] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.GIFT_BASKET, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.GIFT_BASKET] = basket_orders
            return aggregate_orders

    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.GIFT_BASKET not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.GIFT_BASKET]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD]["spread_std_window"]
        ):
            return None
        elif (
            len(spread_data["spread_history"])
            > self.params[Product.SPREAD]["spread_std_window"]
        ):
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[Product.SPREAD]["default_spread_mean"]
        ) / spread_std

        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(
                    -self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(
                    self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None
    # Returns buy_order_volume, sell_order_volume
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

            if best_ask <= fair_value - take_width:
                quantity = min(
                    best_ask_amount, position_limit - position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):

        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
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
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

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
            # Aggregate volume from all buy orders with price greater than fair_for_ask
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
            # Aggregate volume from all sell orders with price lower than fair_for_bid
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

    def get_synthetic_basket_order_depth2(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        JAMS_PER_BASKET = BASKET_WEIGHTS2[Product.JAMS]
        CROISSANTS_PER_BASKET = BASKET_WEIGHTS2[Product.CROISSANTS]
        DJEMBES_PER_BASKET = BASKET_WEIGHTS2[Product.DJEMBES]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        JAMS_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        JAMS_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        CROISSANTS_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        CROISSANTS_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        DJEMBES_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        DJEMBES_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            JAMS_best_bid * JAMS_PER_BASKET
            + CROISSANTS_best_bid * CROISSANTS_PER_BASKET
            + DJEMBES_best_bid * DJEMBES_PER_BASKET
        )
        implied_ask = (
            JAMS_best_ask * JAMS_PER_BASKET
            + CROISSANTS_best_ask * CROISSANTS_PER_BASKET
            + DJEMBES_best_ask * DJEMBES_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            JAMS_bid_volume = (
                order_depths[Product.JAMS].buy_orders[JAMS_best_bid]
                // JAMS_PER_BASKET
            )
            CROISSANTS_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[CROISSANTS_best_bid]
                // CROISSANTS_PER_BASKET
            )
            DJEMBES_bid_volume = (
                order_depths[Product.DJEMBES].buy_orders[DJEMBES_best_bid]
                // DJEMBES_PER_BASKET
            )
            implied_bid_volume = min(
                JAMS_bid_volume, CROISSANTS_bid_volume, DJEMBES_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            JAMS_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[JAMS_best_ask]
                // JAMS_PER_BASKET
            )
            CROISSANTS_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[CROISSANTS_best_ask]
                // CROISSANTS_PER_BASKET
            )
            DJEMBES_ask_volume = (
                -order_depths[Product.DJEMBES].sell_orders[DJEMBES_best_ask]
                // DJEMBES_PER_BASKET
            )
            implied_ask_volume = min(
                JAMS_ask_volume, CROISSANTS_ask_volume, DJEMBES_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket_orders2(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.JAMS: [],
            Product.CROISSANTS: [],
            Product.DJEMBES: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth2(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                JAMS_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                CROISSANTS_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                DJEMBES_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                JAMS_price = max(order_depths[Product.JAMS].buy_orders.keys())
                CROISSANTS_price = max(
                    order_depths[Product.CROISSANTS].buy_orders.keys()
                )
                DJEMBES_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            JAMS_order = Order(
                Product.JAMS,
                JAMS_price,
                quantity * BASKET_WEIGHTS2[Product.JAMS],
            )
            CROISSANTS_order = Order(
                Product.CROISSANTS,
                CROISSANTS_price,
                quantity * BASKET_WEIGHTS2[Product.CROISSANTS],
            )
            DJEMBES_order = Order(
                Product.DJEMBES, DJEMBES_price, quantity * BASKET_WEIGHTS2[Product.DJEMBES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.JAMS].append(JAMS_order)
            component_orders[Product.CROISSANTS].append(CROISSANTS_order)
            component_orders[Product.DJEMBES].append(DJEMBES_order)

        return component_orders

    def execute_spread_orders2(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.GIFT_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket_order_depth2(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.GIFT_BASKET2, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC2, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders2(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.GIFT_BASKET2] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.GIFT_BASKET2, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC2, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders2(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.GIFT_BASKET2] = basket_orders
            return aggregate_orders

    def spread_orders2(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.GIFT_BASKET2 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.GIFT_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket_order_depth2(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD2]["spread_std_window"]
        ):
            return None
        elif (
            len(spread_data["spread_history"])
            > self.params[Product.SPREAD2]["spread_std_window"]
        ):
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[Product.SPREAD2]["default_spread_mean"]
        ) / spread_std

        if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread_orders2(
                    -self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread_orders2(
                    self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None
    
    def make_amethyst_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        volume_limit: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        baaf = min(
            [
                price
                for price in order_depth.sell_orders.keys()
                if price > fair_value + 1
            ]
        )
        bbbf = max(
            [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        )

        if baaf <= fair_value + 2:
            if position <= volume_limit:
                baaf = fair_value + 3  # still want edge 2 if position is not a concern

        if bbbf >= fair_value - 2:
            if position >= -volume_limit:
                bbbf = fair_value - 3  # still want edge 2 if position is not a concern

        buy_order_volume, sell_order_volume = self.market_make(
            Product.RAINFOREST_RESIN,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

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

        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
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






    def MultiTrader_orders(self, state: TradingState):
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
                            if best_ask and best_ask <= entry * 0.975:  # 1.5% drop
                                qty = min(-pos, limit + pos)
                                if qty > 0:
                                    orders[product].append(Order(product, best_ask, qty))
                                    self.logger.info(f"Exit short {product} (Caesar): {qty} at {best_ask}")
                                    del self.entry_prices[(p, ts)]
                            elif best_ask and best_ask >= entry * 1.02:  # 1% rise (stop-loss)
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
                        if p == product and pos > 0 : #and best_bid and best_bid >= entry * 1.03:
                            if best_bid and best_bid >= entry * 1.03: 
                                qty = min(pos, limit - pos)
                                if qty > 0:
                                    orders[product].append(Order(product, best_bid, -qty))
                                    self.logger.info(f"Exit long {product} (Charlie): {qty} at {best_bid}")
                                    del self.entry_prices[(p, ts)]
                            elif best_bid and best_bid <= entry * 0.99:  # 1% drop (stop-loss)
                                qty = min(pos, limit - pos)
                                if qty > 0:
                                    orders[product].append(Order(product, best_bid, -qty))
                                    self.logger.info(f"Stop-loss long {product} (Charlie): {qty} at {best_bid}")
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












    def run(self, state: TradingState):
        Multi_Orders, a, b = self.MultiTrader_orders(state)

        traderObject = {}

        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}
        conversions = 0


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
            amethyst_make_orders, _, _ = self.make_amethyst_orders(
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                amethyst_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["volume_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                amethyst_take_orders + amethyst_clear_orders + amethyst_make_orders
            )


        if Product.MACARONS in self.params and Product.MACARONS in state.order_depths:
            if Product.MACARONS not in traderObject:
                traderObject[Product.MACARONS] = {
                    "curr_edge": self.params[Product.MACARONS]["init_make_edge"],
                    "volume_history": [],
                    "optimized": False
                }
            macarons_position = (
                state.position[Product.MACARONS]
                if Product.MACARONS in state.position
                else 0
            )

            conversions = self.macarons_arb_clear(macarons_position)

            adap_edge = self.macarons_adap_edge(
                state.timestamp,
                traderObject[Product.MACARONS]["curr_edge"],
                macarons_position,
                traderObject
            )

            macarons_position = 0  # Reset position after conversions

            macarons_take_orders, buy_order_volume, sell_order_volume = self.macarons_arb_take(
                state.order_depths[Product.MACARONS],
                state.observations.conversionObservations[Product.MACARONS],
                adap_edge,
                macarons_position
            )

            macarons_make_orders, _, _ = self.macarons_arb_make(
                state.order_depths[Product.MACARONS],
                state.observations.conversionObservations[Product.MACARONS],
                macarons_position,
                adap_edge,
                buy_order_volume,
                sell_order_volume
            )

            result[Product.MACARONS] = macarons_take_orders + macarons_make_orders
        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position = (
            state.position[Product.GIFT_BASKET]
            if Product.GIFT_BASKET in state.position
            else 0
        )
        spread_orders = self.spread_orders(
            state.order_depths,
            Product.GIFT_BASKET,
            basket_position,
            traderObject[Product.SPREAD],
        )

        if spread_orders != None:
            #+result[Product.JAMS] = spread_orders[Product.JAMS]
            #result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
            #result[Product.DJEMBES] = spread_orders[Product.DJEMBES]
            result[Product.GIFT_BASKET] = spread_orders[Product.GIFT_BASKET]
            
        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position2 = (
            state.position[Product.GIFT_BASKET2]
            if Product.GIFT_BASKET2 in state.position
            else 0
        )
        spread_orders2 = self.spread_orders2(
            state.order_depths,
            Product.GIFT_BASKET2,
            basket_position2,
            traderObject[Product.SPREAD2],
        )

        if spread_orders2 != None:
            #result[Product.JAMS] = spread_orders[Product.JAMS]
            #result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
            #result[Product.DJEMBES] = spread_orders[Product.DJEMBES]
            result[Product.GIFT_BASKET2] = spread_orders2[Product.GIFT_BASKET2]

        traderData = jsonpickle.encode(traderObject)

        # Il faut merge les deux dictionnaires
        merged_dict = {}

        # Merge the two dictionaries
        for key in set(result) | set(Multi_Orders):
            merged_dict[key] = result.get(key, []) + Multi_Orders.get(key, [])

        return merged_dict, conversions, traderData