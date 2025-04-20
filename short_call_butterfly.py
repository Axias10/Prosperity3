from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import jsonpickle
import numpy as np
import math
from statistics import NormalDist


class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"


PARAMS = {   
    Product.VOLCANIC_ROCK: {
        "position_limit": 50
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "strike": 9500,
        "position_limit": 50,
        "max_hold_time": 5
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "strike": 9750,
        "position_limit": 50,
        "max_hold_time": 5
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "strike": 10000,
        "position_limit": 50,
        "max_hold_time": 5
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "strike": 10250,
        "position_limit": 50,
        "max_hold_time": 5
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "strike": 10500,
        "position_limit": 50,
        "max_hold_time": 5
    }
}


def black_scholes_call(spot, strike, time_to_expiry, volatility):
    """Calculate Black-Scholes call option price."""
    if time_to_expiry <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
        return max(0, spot - strike)
    try:
        d1 = (
            math.log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price
    except (ValueError, ZeroDivisionError):
        return max(0, spot - strike)


def implied_volatility(call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
    """Estimate implied volatility using bisection method."""
    if call_price <= 0 or spot <= 0 or strike <= 0 or time_to_expiry <= 0:
        return 0
    low_vol = 0.01
    high_vol = 1.0
    volatility = (low_vol + high_vol) / 2.0
    for _ in range(max_iterations):
        estimated_price = black_scholes_call(spot, strike, time_to_expiry, volatility)
        diff = estimated_price - call_price
        if abs(diff) < tolerance:
            break
        elif diff > 0:
            high_vol = volatility
        else:
            low_vol = volatility
        volatility = (low_vol + high_vol) / 2.0
    return volatility


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.total_pnl = 0
        self.MAX_PNL_LOSS = -500

        self.LIMIT = {
            Product.VOLCANIC_ROCK: 50,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 50,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 50,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 50,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 50,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 50
        }

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
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if not prevent_adverse or (prevent_adverse and best_ask_amount <= adverse_volume):
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
            if not prevent_adverse or (prevent_adverse and best_bid_amount <= adverse_volume):
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[product]
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = fair_value - clear_width
        fair_for_ask = fair_value + clear_width
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, round(fair_for_ask), -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, round(fair_for_bid), abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        prevent_adverse: bool = False,
        soft_position_limit: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[product]
        disregard_edge = disregard_edge if disregard_edge else default_edge
        join_edge = join_edge if join_edge else default_edge
        aaf = [price for price in order_depth.sell_orders.keys() if price >= round(fair_value + join_edge)]
        bbf = [price for price in order_depth.buy_orders.keys() if price <= round(fair_value - join_edge)]
        baaf = min(aaf) if len(aaf) > 0 else round(fair_value + default_edge)
        bbbf = max(bbf) if len(bbf) > 0 else round(fair_value - default_edge)
        if prevent_adverse and soft_position_limit != 0:
            if position <= soft_position_limit:
                baaf = fair_value + default_edge
            if position >= -soft_position_limit:
                bbbf = fair_value - default_edge
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bbbf + disregard_edge), buy_quantity))
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(baaf - disregard_edge), -sell_quantity))
        return orders, buy_order_volume, sell_order_volume

    def calculate_m_t(self, S, K, TTE):
        """Calculate m_t = log(K/S) / sqrt(TTE)."""
        if TTE <= 0 or S <= 0:
            return 0
        try:
            return math.log(K / S) / math.sqrt(TTE)
        except (ValueError, ZeroDivisionError):
            return 0

    def fit_parabolic_curve(self, m_t_list, v_t_list):
        if len(m_t_list) < 2 or len(v_t_list) < 2:
            return lambda m: 0.2  # Valeur par défaut si pas assez de points

        # Ajuster une parabole (ax^2 + bx + c) sur les points (m_t, v_t)
        # Ici, on utilise uniquement m_t[0] et m_t[2] (low et high)
        try:
            # Ajustement polynomial de degré 2
            coeffs = np.polyfit(m_t_list, v_t_list, 2)  # [a, b, c]
            return lambda m: coeffs[0] * m**2 + coeffs[1] * m + coeffs[2]
        except:
            return lambda m: 0.2  # En cas d'erreur, retourner une constante

    def short_call_butterfly_strategy(
        self,
        underlying_product: str,
        option_low: str,
        option_mid: str,
        option_high: str,
        underlying_order_depth: OrderDepth,
        option_low_depth: OrderDepth,
        option_mid_depth: OrderDepth,
        option_high_depth: OrderDepth,
        traderObject: dict,
        positions: dict,
        timestamp: int
    ) -> dict:
        orders = {
            option_low: [],
            option_mid: [],
            option_high: []
        }

        # Skip if not enough orders
        if (not underlying_order_depth.sell_orders or not underlying_order_depth.buy_orders or
            not option_low_depth.sell_orders or not option_low_depth.buy_orders or
            not option_mid_depth.sell_orders or not option_mid_depth.buy_orders or
            not option_high_depth.sell_orders or not option_high_depth.buy_orders):
            return orders

        # Calculate underlying mid-price
        best_ask_underlying = min(underlying_order_depth.sell_orders.keys())
        best_bid_underlying = max(underlying_order_depth.buy_orders.keys())
        underlying_mid = (best_ask_underlying + best_bid_underlying) / 2

        # Estimate TTE (7 days total, adjusted for 100000 timestamps)
        round_number = timestamp / 14286  # 1 jour ≈ 14286 timestamps
        TTE = max(0, (7 - round_number) / 365)  # Convert to years for Black-Scholes
        print(f"TTE: {TTE}, Round Number: {round_number}")

        # Calculate implied volatilities and m_t
        vouchers = [
            (option_low, option_low_depth, self.params[option_low]["strike"]),
            (option_mid, option_mid_depth, self.params[option_mid]["strike"]),
            (option_high, option_high_depth, self.params[option_high]["strike"])
        ]
        m_t_list = []
        v_t_list = []
        mid_prices = {}
        iv_values = {}
        for product, depth, K in vouchers:
            best_ask = min(depth.sell_orders.keys())
            best_bid = max(depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            mid_prices[product] = mid_price
            if TTE > 0 and mid_price > 0:
                iv = implied_volatility(mid_price, underlying_mid, K, TTE)
                m_t = self.calculate_m_t(underlying_mid, K, TTE)
                m_t_list.append(m_t)
                v_t_list.append(iv)
                iv_values[product] = iv
            else:
                m_t_list.append(0)
                v_t_list.append(0)
                iv_values[product] = 0

        # Fit parabolic curve excluding iv_mid, with fallback to linear interpolation
        m_t_fit = [m_t_list[0], m_t_list[2]]  # m_t pour low et high
        v_t_fit = [v_t_list[0], v_t_list[2]]  # iv pour low et high
        if max(m_t_list) - min(m_t_list) < 1e-5:  # Si les m_t sont presque identiques
            print("Warning: m_t values too close, using linear interpolation")
            def fitted_vol(m):
                m_low, m_high = m_t_list[0], m_t_list[2]
                v_low, v_high = v_t_list[0], v_t_list[2]
                if m_low == m_high:
                    return (v_low + v_high) / 2
                slope = (v_high - v_low) / (m_high - m_low)
                return v_low + slope * (m - m_low)
        else:
            try:
                coeffs = np.polyfit(m_t_fit, v_t_fit, 2)  # [a, b, c]
                def fitted_vol(m):
                    return coeffs[0] * m**2 + coeffs[1] * m + coeffs[2]
            except np.RankWarning:
                print("Polyfit failed, using linear interpolation")
                def fitted_vol(m):
                    m_low, m_high = m_t_list[0], m_t_list[2]
                    v_low, v_high = v_t_list[0], v_t_list[2]
                    if m_low == m_high:
                        return (v_low + v_high) / 2
                    slope = (v_high - v_low) / (m_high - m_low)
                    return v_low + slope * (m - m_low)

        # Calculer le base IV (v_t(m_t=0))
        base_iv = fitted_vol(0)

        # Get current positions
        position_low = positions.get(option_low, 0)
        position_mid = positions.get(option_mid, 0)
        position_high = positions.get(option_high, 0)

        # Position tracking
        position_key = f"{option_mid}_butterfly_position"
        entry_time_key = f"{option_mid}_butterfly_entry_time"
        entry_cost_key = f"{option_mid}_butterfly_entry_cost"
        last_trade_time_key = f"{option_mid}_last_trade_time"
        butterfly_position = traderObject.get(position_key, 0)

        # Manage existing position
        if butterfly_position != 0:
            entry_time = traderObject.get(entry_time_key, 0)
            entry_cost = traderObject.get(entry_cost_key, 0)
            hold_time =  round_number - entry_time

            # Calculate current position value
            best_bid_low = max(option_low_depth.buy_orders.keys())
            best_ask_mid = min(option_mid_depth.sell_orders.keys())
            best_bid_high = max(option_high_depth.buy_orders.keys())
            current_value = (position_low * best_bid_low) + (position_mid * best_ask_mid) + (position_high * best_bid_high)
            position_pnl = current_value - entry_cost

            # Exit conditions
            max_hold_time = 2000  # Réduit à 2000 timestamps (environ 1 jour)
            profit_target = entry_cost * 0.15
            loss_limit = entry_cost * -0.2
            iv_spread = abs(fitted_vol(self.calculate_m_t(underlying_mid, self.params[option_mid]["strike"], TTE)) - iv_values[option_mid])

            # Condition de sortie basée sur le base IV
            base_iv_current = fitted_vol(0)
            if abs(base_iv_current - traderObject.get("entry_base_iv", base_iv_current)) > 0.03:  # Réduit à 0.03
                print(f"Exiting Trade: Base IV changed significantly (from {traderObject.get('entry_base_iv')} to {base_iv_current})")
                if position_low > 0:
                    orders[option_low].append(Order(option_low, best_bid_low, -position_low))
                if position_mid < 0:
                    orders[option_mid].append(Order(option_mid, best_ask_mid, -position_mid))
                if position_high > 0:
                    orders[option_high].append(Order(option_high, best_bid_high, -position_high))
                traderObject[position_key] = 0
                traderObject.pop(entry_time_key, None)
                traderObject.pop(entry_cost_key, None)
                traderObject.pop("entry_base_iv", None)
                return orders

            if (hold_time >= max_hold_time or
                position_pnl >= profit_target or
                position_pnl <= loss_limit or
                iv_spread > 0.03):
                if position_low > 0:
                    orders[option_low].append(Order(option_low, best_bid_low, -position_low))
                if position_mid < 0:
                    orders[option_mid].append(Order(option_mid, best_ask_mid, -position_mid))
                if position_high > 0:
                    orders[option_high].append(Order(option_high, best_bid_high, -position_high))
                traderObject[position_key] = 0
                traderObject.pop(entry_time_key, None)
                traderObject.pop(entry_cost_key, None)
                traderObject.pop("entry_base_iv", None)
            return orders

        # Check for new butterfly position
        position_limit_low = self.params[option_low]["position_limit"]
        position_limit_mid = self.params[option_mid]["position_limit"]
        position_limit_high = self.params[option_high]["position_limit"]

        print(f"Positions: low={position_low}, mid={position_mid}, high={position_high}")
        print(f"Position Limits: low={position_limit_low}, mid={position_limit_mid}, high={position_limit_high}")

        # Vérifier la période de repos
        last_trade_time = traderObject.get(last_trade_time_key, -1000)
        if timestamp - last_trade_time < 400:  # Réduit à 400 timestamps
            return orders

        if (position_low < position_limit_low and
            position_mid > -position_limit_mid and
            position_high < position_limit_high):

            # Get best prices
            best_ask_low = min(option_low_depth.sell_orders.keys())
            best_bid_mid = max(option_mid_depth.buy_orders.keys())
            best_ask_high = min(option_high_depth.sell_orders.keys())

            # Calculate cost
            cost = (2 * best_bid_mid) - best_ask_low - best_ask_high
            print(f"Cost: {cost}, Underlying Mid: {underlying_mid}")

            # Calculate IV spread relative to fitted curve
            m_t_mid = self.calculate_m_t(underlying_mid, self.params[option_mid]["strike"], TTE)
            fitted_iv_mid = fitted_vol(m_t_mid)
            iv_low = iv_values[option_low]
            iv_mid = iv_values[option_mid]
            iv_high = iv_values[option_high]

            print(f"IVs: low={iv_low}, mid={iv_mid}, high={iv_high}, fitted_mid={fitted_iv_mid}")

            # Ajuster le seuil de volatilité
            volatility_threshold = 0.015 * (base_iv / 0.15)
            volatility_threshold = max(0.008, min(0.025, volatility_threshold))

            # Assouplir la plage de base IV
            if base_iv < 0.06 or base_iv > 0.3:
                print(f"Trade Rejected: Base IV out of range (base_iv={base_iv})")
                return orders

            # Enter if middle strike IV is overvalued and cost is favorable
            if (iv_mid > fitted_iv_mid + volatility_threshold and
                -75 <= cost <= 75 and  # Élargie à [-75, 75]
                9500 <= underlying_mid <= 10500):
                quantity = 3  # Augmenté à 3
                if (position_low + quantity <= position_limit_low and
                    position_mid - 2 * quantity >= -position_limit_mid and
                    position_high + quantity <= position_limit_high):
                    orders[option_low].append(Order(option_low, best_ask_low, quantity))
                    orders[option_mid].append(Order(option_mid, best_bid_mid, -2 * quantity))
                    orders[option_high].append(Order(option_high, best_ask_high, quantity))

                    traderObject[position_key] = quantity
                    traderObject[entry_time_key] = round_number
                    traderObject[entry_cost_key] = -cost
                    traderObject[last_trade_time_key] = timestamp
                    traderObject["entry_base_iv"] = base_iv
                    print(f"Trade Initiated: {option_low} at {best_ask_low}, {option_mid} at {best_bid_mid}, {option_high} at {best_ask_high}")
            else:
                if not (iv_mid > fitted_iv_mid + volatility_threshold):
                    print(f"Trade Rejected: IV condition failed (iv_mid={iv_mid}, fitted_mid={fitted_iv_mid})")
                if not (-75 <= cost <= 75):
                    print(f"Trade Rejected: Cost condition failed (cost={cost})")
                if not (9500 <= underlying_mid <= 10500):
                    print(f"Trade Rejected: Underlying condition failed (underlying_mid={underlying_mid})")

        return orders

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        # Update total PnL
        for product, trades in state.market_trades.items():
            for trade in trades:
                if trade.buyer == "SUBMISSION":
                    self.total_pnl -= trade.price * trade.quantity
                if trade.seller == "SUBMISSION":
                    self.total_pnl += trade.price * trade.quantity

        if self.total_pnl < self.MAX_PNL_LOSS:
            return {}, 0, jsonpickle.encode(traderObject)

        result = {}

        if (Product.VOLCANIC_ROCK in state.order_depths and
            Product.VOLCANIC_ROCK_VOUCHER_9500 in state.order_depths and
            Product.VOLCANIC_ROCK_VOUCHER_9750 in state.order_depths and
            Product.VOLCANIC_ROCK_VOUCHER_10000 in state.order_depths and
            Product.VOLCANIC_ROCK_VOUCHER_10250 in state.order_depths and
            Product.VOLCANIC_ROCK_VOUCHER_10500 in state.order_depths):
            positions = {
                Product.VOLCANIC_ROCK: state.position.get(Product.VOLCANIC_ROCK, 0),
                Product.VOLCANIC_ROCK_VOUCHER_9500: state.position.get(Product.VOLCANIC_ROCK_VOUCHER_9500, 0),
                Product.VOLCANIC_ROCK_VOUCHER_9750: state.position.get(Product.VOLCANIC_ROCK_VOUCHER_9750, 0),
                Product.VOLCANIC_ROCK_VOUCHER_10000: state.position.get(Product.VOLCANIC_ROCK_VOUCHER_10000, 0),
                Product.VOLCANIC_ROCK_VOUCHER_10250: state.position.get(Product.VOLCANIC_ROCK_VOUCHER_10250, 0),
                Product.VOLCANIC_ROCK_VOUCHER_10500: state.position.get(Product.VOLCANIC_ROCK_VOUCHER_10500, 0)
            }
            butterfly_orders_1 = self.short_call_butterfly_strategy(
                Product.VOLCANIC_ROCK,
                Product.VOLCANIC_ROCK_VOUCHER_9500,
                Product.VOLCANIC_ROCK_VOUCHER_9750,
                Product.VOLCANIC_ROCK_VOUCHER_10000,
                state.order_depths[Product.VOLCANIC_ROCK],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9500],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9750],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10000],
                traderObject,
                positions,
                state.timestamp
            )
            butterfly_orders_2 = self.short_call_butterfly_strategy(
                Product.VOLCANIC_ROCK,
                Product.VOLCANIC_ROCK_VOUCHER_9750,
                Product.VOLCANIC_ROCK_VOUCHER_10000,
                Product.VOLCANIC_ROCK_VOUCHER_10250,
                state.order_depths[Product.VOLCANIC_ROCK],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9750],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10000],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10250],
                traderObject,
                positions,
                state.timestamp
            )
            butterfly_orders_3 = self.short_call_butterfly_strategy(
                Product.VOLCANIC_ROCK,
                Product.VOLCANIC_ROCK_VOUCHER_10000,
                Product.VOLCANIC_ROCK_VOUCHER_10250,
                Product.VOLCANIC_ROCK_VOUCHER_10500,
                state.order_depths[Product.VOLCANIC_ROCK],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10000],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10250],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10500],
                traderObject,
                positions,
                state.timestamp
            )
            result[Product.VOLCANIC_ROCK_VOUCHER_9500] = butterfly_orders_1[Product.VOLCANIC_ROCK_VOUCHER_9500]
            result[Product.VOLCANIC_ROCK_VOUCHER_9750] = (butterfly_orders_1[Product.VOLCANIC_ROCK_VOUCHER_9750] +
                                                        butterfly_orders_2[Product.VOLCANIC_ROCK_VOUCHER_9750])
            result[Product.VOLCANIC_ROCK_VOUCHER_10000] = (butterfly_orders_1[Product.VOLCANIC_ROCK_VOUCHER_10000] +
                                                         butterfly_orders_2[Product.VOLCANIC_ROCK_VOUCHER_10000] +
                                                         butterfly_orders_3[Product.VOLCANIC_ROCK_VOUCHER_10000])
            result[Product.VOLCANIC_ROCK_VOUCHER_10250] = (butterfly_orders_2[Product.VOLCANIC_ROCK_VOUCHER_10250] +
                                                         butterfly_orders_3[Product.VOLCANIC_ROCK_VOUCHER_10250])
            result[Product.VOLCANIC_ROCK_VOUCHER_10500] = butterfly_orders_3[Product.VOLCANIC_ROCK_VOUCHER_10500]

        conversions = 0
        traderData = jsonpickle.encode(traderObject)
        return result, conversions, traderData