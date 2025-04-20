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
    JAMS = "JAMS"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"  # Added option product

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 1,
        "disregard_edge": 0.25,
        "join_edge": 3,
        "default_edge": 1,
        "soft_position_limit": 15,
    },
    Product.SQUID_INK: {
        "take_width": 3,
        "clear_width": 3,
        "prevent_adverse": True,
        "adverse_volume": 5,
        "reversion_beta": -0.3,
        "disregard_edge": 2,
        "join_edge": 1,
        "default_edge": 3,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "reversion_beta": -0.3,
        "disregard_edge": 2,
        "join_edge": 1,
        "default_edge": 5,
    },
    Product.CROISSANTS: {
        "take_width": 1.5,
        "clear_width": 1.0,
        "mean_reversion_strength": -0.4,
        "history_window": 20,
        "sinc_peak_threshold": 0.5,
        "sinc_center": 10000,
        "sinc_width": 750,
        "position_limit": 40,
        "buy_threshold": 0.3,
        "sell_threshold": 0.2,
        "max_spread": 10,
    },
    Product.JAMS: {
        "take_width": 1.5,
        "clear_width": 1.0,
        "mean_reversion_strength": -0.4,
        "history_window": 20,
        "sinc_peak_threshold": 0.5,
        "sinc_center": 10000,
        "sinc_width": 750,
        "position_limit": 40,
        "buy_threshold": 0.3,
        "sell_threshold": 0.2,
        "max_spread": 10,
    },
    Product.VOLCANIC_ROCK: {
        "ma_period": 75,
        "bb_std": 2.5,
        "atr_period": 14,
        "position_limit": 50,
        "risk_per_trade": 0.75,
        "stop_loss_atr": 2,
        "profit_target": 0.5,
        "max_hold_time": 5,
        "min_reversal_strength": 0.5,
        "vol_adjust_factor": 0.8,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {  # Parameters for the option
        "strike": 10500,  # Strike price of the option
        "position_limit": 10,  # Limit on option positions
        "min_sell_premium": 300,  # Minimum premium to sell the option
        "stop_loss_underlying": 10250,  # Stop-loss on VOLCANIC_ROCK price
        "max_hold_time": 5,  # Maximum holding time for short position
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
            Product.JAMS: 40,
            Product.VOLCANIC_ROCK: 50,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 10  # Added position limit for the option
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

    def calculate_bollinger_bands(self, prices, ma_period=20, bb_std=2.0):
        """Calculate Bollinger Bands for a given price series"""
        if len(prices) < ma_period:
            return None, None, None
        
        # Calculate moving average
        ma = np.mean(prices[-ma_period:])
        
        # Calculate standard deviation
        std = np.std(prices[-ma_period:])
        
        # Calculate upper and lower bands
        upper_band = ma + (bb_std * std)
        lower_band = ma - (bb_std * std)
        
        return ma, upper_band, lower_band
    
    def calculate_atr(self, high_prices, low_prices, close_prices, period=14):
        """Calculate Average True Range for volatility measurement"""
        if len(high_prices) < period + 1:
            return None
        
        # Create lists to store true ranges
        true_ranges = []
        
        # Calculate True Range for each period
        for i in range(1, len(close_prices)):
            high = high_prices[i]
            low = low_prices[i]
            prev_close = close_prices[i-1]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        # Calculate ATR as the average of the true ranges
        if len(true_ranges) < period:
            return np.mean(true_ranges)
        
        return np.mean(true_ranges[-period:])
    
    def detect_reversal(self, prices, current_price, upper_band, lower_band, min_strength=0.3):
        """Detect potential price reversals at Bollinger Band boundaries"""
        if len(prices) < 3:
            return 0
        
        # Calculate recent price changes
        recent_changes = [prices[i] - prices[i-1] for i in range(len(prices)-2, len(prices))]
        avg_change = np.mean(np.abs(recent_changes))
        
        # Check for lower band touch with reversal signs
        if current_price <= lower_band:
            # Looking for positive price change after touching lower band
            if prices[-1] > prices[-2] and abs(prices[-1] - prices[-2]) > min_strength * avg_change:
                return 1  # Buy signal
        
        # Check for upper band touch with reversal signs
        if current_price >= upper_band:
            # Looking for negative price change after touching upper band
            if prices[-1] < prices[-2] and abs(prices[-1] - prices[-2]) > min_strength * avg_change:
                return -1  # Sell signal
        
        return 0  # No signal
    
    def adjust_position_size(self, base_size, current_atr, avg_atr, vol_adjust_factor=0.8):
        """Adjust position size based on current volatility relative to average volatility"""
        if avg_atr is None or current_atr is None or avg_atr == 0:
            return base_size
        
        # Calculate volatility ratio
        vol_ratio = current_atr / avg_atr
        
        # Adjust position size inversely with volatility
        adjusted_size = base_size * (vol_adjust_factor / vol_ratio)
        
        # Ensure minimum position size
        return max(1, round(adjusted_size))
    
    def check_time_based_exit(self, entry_time, current_time, max_hold_time=5):
        """Check if position should be exited based on holding time"""
        if entry_time is None:
            return False
        
        hold_time = current_time - entry_time
        return hold_time >= max_hold_time
    
    def volcanic_rock_strategy(
        self,
        product: str,
        order_depth: OrderDepth,
        traderObject,
        position: int,
        timestamp: int
    ) -> List[Order]:
        """Implement Bollinger Band Mean-Reversion Volatility Strategy for VOLCANIC_ROCK"""
        orders = []
        
        # Skip if not enough orders in the book
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return orders
            
        # Initialize price history if not exists
        price_history_key = f"{product}_price_history"
        if price_history_key not in traderObject:
            traderObject[price_history_key] = []
            
        # Initialize high, low, close histories
        high_history_key = f"{product}_high_history"
        low_history_key = f"{product}_low_history"
        close_history_key = f"{product}_close_history"
        
        if high_history_key not in traderObject:
            traderObject[high_history_key] = []
        if low_history_key not in traderObject:
            traderObject[low_history_key] = []
        if close_history_key not in traderObject:
            traderObject[close_history_key] = []
            
        # Initialize position tracking
        position_key = f"{product}_position"
        entry_time_key = f"{product}_entry_time"
        entry_price_key = f"{product}_entry_price"
        stop_loss_key = f"{product}_stop_loss"
        profit_target_key = f"{product}_profit_target"
        
        # Get current market prices
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2
        
        # Update price history
        traderObject[price_history_key].append(mid_price)
        
        # Update high/low/close prices for ATR calculation
        traderObject[high_history_key].append(best_ask)  # Using ask as high
        traderObject[low_history_key].append(best_bid)   # Using bid as low
        traderObject[close_history_key].append(mid_price)  # Using mid as close
        
        # Limit history length
        ma_period = self.params[product]["ma_period"]
        history_limit = max(ma_period * 2, 50)  # Keep enough history for calculations
        
        if len(traderObject[price_history_key]) > history_limit:
            traderObject[price_history_key] = traderObject[price_history_key][-history_limit:]
            traderObject[high_history_key] = traderObject[high_history_key][-history_limit:]
            traderObject[low_history_key] = traderObject[low_history_key][-history_limit:]
            traderObject[close_history_key] = traderObject[close_history_key][-history_limit:]
        
        # Calculate Bollinger Bands
        ma, upper_band, lower_band = self.calculate_bollinger_bands(
            traderObject[price_history_key],
            self.params[product]["ma_period"],
            self.params[product]["bb_std"]
        )
        
        # Calculate ATR for volatility measurement
        current_atr = self.calculate_atr(
            traderObject[high_history_key],
            traderObject[low_history_key],
            traderObject[close_history_key],
            self.params[product]["atr_period"]
        )
        
        # Initialize average ATR if not exists
        avg_atr_key = f"{product}_avg_atr"
        if avg_atr_key not in traderObject and current_atr is not None:
            traderObject[avg_atr_key] = current_atr
        elif current_atr is not None:
            # Update average ATR with exponential smoothing
            traderObject[avg_atr_key] = 0.9 * traderObject[avg_atr_key] + 0.1 * current_atr
        
        # Skip strategy execution if we don't have enough data yet
        if ma is None or current_atr is None:
            return orders
        
        # Store Bollinger Bands values for reference
        traderObject[f"{product}_ma"] = ma
        traderObject[f"{product}_upper_band"] = upper_band
        traderObject[f"{product}_lower_band"] = lower_band
        traderObject[f"{product}_current_atr"] = current_atr
        
        # Detect reversal signals
        reversal_signal = self.detect_reversal(
            traderObject[price_history_key],
            mid_price,
            upper_band,
            lower_band,
            self.params[product]["min_reversal_strength"]
        )
        
        # Get position limits
        position_limit = self.params[product]["position_limit"]
        
        # Check if we need to manage existing position
        current_position = traderObject.get(position_key, 0)
        
        # Position management (check stop loss, profit target, time-based exit)
        if current_position != 0:
            entry_time = traderObject.get(entry_time_key)
            entry_price = traderObject.get(entry_price_key)
            stop_loss = traderObject.get(stop_loss_key)
            profit_target = traderObject.get(profit_target_key)
            
            # Check stop loss
            if current_position > 0 and best_bid <= stop_loss:
                # Close long position at stop loss
                orders.append(Order(product, best_bid, -current_position))
                traderObject[position_key] = 0
                # Clear other position tracking data
                traderObject.pop(entry_time_key, None)
                traderObject.pop(entry_price_key, None)
                traderObject.pop(stop_loss_key, None)
                traderObject.pop(profit_target_key, None)
                
            elif current_position < 0 and best_ask >= stop_loss:
                # Close short position at stop loss
                orders.append(Order(product, best_ask, -current_position))
                traderObject[position_key] = 0
                # Clear other position tracking data
                traderObject.pop(entry_time_key, None)
                traderObject.pop(entry_price_key, None)
                traderObject.pop(stop_loss_key, None)
                traderObject.pop(profit_target_key, None)
                
            # Check profit target (moving average)
            elif (current_position > 0 and best_bid >= profit_target) or \
                 (current_position < 0 and best_ask <= profit_target):
                # Take profit
                price = best_bid if current_position > 0 else best_ask
                orders.append(Order(product, price, -current_position))
                traderObject[position_key] = 0
                # Clear other position tracking data
                traderObject.pop(entry_time_key, None)
                traderObject.pop(entry_price_key, None)
                traderObject.pop(stop_loss_key, None)
                traderObject.pop(profit_target_key, None)
                
            # Check time-based exit
            elif self.check_time_based_exit(
                entry_time, 
                timestamp, 
                self.params[product]["max_hold_time"]
            ):
                # Exit position based on time
                price = best_bid if current_position > 0 else best_ask
                orders.append(Order(product, price, -current_position))
                traderObject[position_key] = 0
                # Clear other position tracking data
                traderObject.pop(entry_time_key, None)
                traderObject.pop(entry_price_key, None)
                traderObject.pop(stop_loss_key, None)
                traderObject.pop(profit_target_key, None)
        
        # Check for new entry signals if no current position
        if traderObject.get(position_key, 0) == 0:
            # Calculate base position size (account for current inventory position)
            base_size = position_limit * self.params[product]["risk_per_trade"]
            
            # Adjust position size based on volatility
            adjusted_size = self.adjust_position_size(
                base_size,
                current_atr,
                traderObject[avg_atr_key],
                self.params[product]["vol_adjust_factor"]
            )
            
            # Ensure adjusted size doesn't exceed position limits
            adjusted_size = min(adjusted_size, position_limit - abs(position))
            
            if reversal_signal == 1 and adjusted_size > 0:  # Buy signal
                # Long entry at lower Bollinger Band
                entry_price = best_ask
                stop_loss = entry_price - (current_atr * self.params[product]["stop_loss_atr"])
                profit_target = entry_price + ((ma - entry_price) * self.params[product]["profit_target"])
                
                # Place buy order
                orders.append(Order(product, entry_price, adjusted_size))
                
                # Update position tracking
                traderObject[position_key] = adjusted_size
                traderObject[entry_time_key] = timestamp
                traderObject[entry_price_key] = entry_price
                traderObject[stop_loss_key] = stop_loss
                traderObject[profit_target_key] = profit_target
                
            elif reversal_signal == -1 and adjusted_size > 0:  # Sell signal
                # Short entry at upper Bollinger Band
                entry_price = best_bid
                stop_loss = entry_price + (current_atr * self.params[product]["stop_loss_atr"])
                profit_target = entry_price - ((entry_price - ma) * self.params[product]["profit_target"])
                
                # Place sell order
                orders.append(Order(product, entry_price, -adjusted_size))
                
                # Update position tracking
                traderObject[position_key] = -adjusted_size
                traderObject[entry_time_key] = timestamp
                traderObject[entry_price_key] = entry_price
                traderObject[stop_loss_key] = stop_loss
                traderObject[profit_target_key] = profit_target
        
        return orders

    def option_sell_strategy(
        self,
        option_product: str,
        option_order_depth: OrderDepth,
        underlying_order_depth: OrderDepth,
        traderObject: dict,
        position: int,
        timestamp: int
    ) -> List[Order]:
        """
        Strategy to sell overvalued options (VOLCANIC_ROCK_VOUCHER_10500).
        Sells the option if its price is high (high premium) and manages the position with a stop-loss
        based on the underlying price.
        """
        orders = []

        # Skip if not enough orders in the book
        if len(option_order_depth.sell_orders) == 0 or len(option_order_depth.buy_orders) == 0:
            return orders
        if len(underlying_order_depth.sell_orders) == 0 or len(underlying_order_depth.buy_orders) == 0:
            return orders

        # Get current market prices for the option
        option_best_ask = min(option_order_depth.sell_orders.keys())
        option_best_bid = max(option_order_depth.buy_orders.keys())

        # Get current market prices for the underlying (VOLCANIC_ROCK)
        underlying_best_ask = min(underlying_order_depth.sell_orders.keys())
        underlying_best_bid = max(underlying_order_depth.buy_orders.keys())
        underlying_mid_price = (underlying_best_ask + underlying_best_bid) / 2

        # Initialize position tracking
        position_key = f"{option_product}_position"
        entry_time_key = f"{option_product}_entry_time"
        entry_price_key = f"{option_product}_entry_price"

        # Get current position
        current_position = traderObject.get(position_key, 0)

        # Position limit for the option
        position_limit = self.params[option_product]["position_limit"]

        # Manage existing short position
        if current_position < 0:
            entry_time = traderObject.get(entry_time_key)
            entry_price = traderObject.get(entry_price_key)

            # Check stop-loss based on underlying price
            stop_loss_underlying = self.params[option_product]["stop_loss_underlying"]
            if underlying_mid_price >= stop_loss_underlying:
                # Close short position by buying back the option
                quantity = abs(current_position)
                orders.append(Order(option_product, option_best_ask, quantity))
                traderObject[position_key] = 0
                traderObject.pop(entry_time_key, None)
                traderObject.pop(entry_price_key, None)
                return orders

            # Check time-based exit
            if self.check_time_based_exit(
                entry_time,
                timestamp,
                self.params[option_product]["max_hold_time"]
            ):
                # Close short position by buying back the option
                quantity = abs(current_position)
                orders.append(Order(option_product, option_best_ask, quantity))
                traderObject[position_key] = 0
                traderObject.pop(entry_time_key, None)
                traderObject.pop(entry_price_key, None)
                return orders

        # Check for new short position if no current position
        if current_position == 0:
            # Calculate intrinsic value of the option
            strike = self.params[option_product]["strike"]
            intrinsic_value = max(underlying_mid_price - strike, 0)
            option_mid_price = (option_best_ask + option_best_bid) / 2
            premium = option_mid_price - intrinsic_value

            # Sell the option if the premium is above the minimum threshold
            min_sell_premium = self.params[option_product]["min_sell_premium"]
            if premium >= min_sell_premium and position > -position_limit:
                quantity = min(1, position_limit + position)  # Sell 1 unit at a time
                orders.append(Order(option_product, option_best_bid, -quantity))
                traderObject[position_key] = -quantity
                traderObject[entry_time_key] = timestamp
                traderObject[entry_price_key] = option_best_bid

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
            
        # Handle VOLCANIC_ROCK with Bollinger Band Mean-Reversion Volatility Strategy
        if Product.VOLCANIC_ROCK in self.params and Product.VOLCANIC_ROCK in state.order_depths:
            volcanic_position = (
                state.position[Product.VOLCANIC_ROCK]
                if Product.VOLCANIC_ROCK in state.position
                else 0
            )
            volcanic_orders = self.volcanic_rock_strategy(
                Product.VOLCANIC_ROCK,
                state.order_depths[Product.VOLCANIC_ROCK],
                traderObject,
                volcanic_position,
                state.timestamp
            )
            result[Product.VOLCANIC_ROCK] = volcanic_orders

        # Handle VOLCANIC_ROCK_VOUCHER_10500 with option selling strategy
        if Product.VOLCANIC_ROCK_VOUCHER_10500 in self.params and Product.VOLCANIC_ROCK_VOUCHER_10500 in state.order_depths:
            option_position = (
                state.position[Product.VOLCANIC_ROCK_VOUCHER_10500]
                if Product.VOLCANIC_ROCK_VOUCHER_10500 in state.position
                else 0
            )
            # Need the underlying order depth for stop-loss
            underlying_order_depth = state.order_depths.get(Product.VOLCANIC_ROCK, None)
            if underlying_order_depth:
                option_orders = self.option_sell_strategy(
                    Product.VOLCANIC_ROCK_VOUCHER_10500,
                    state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10500],
                    underlying_order_depth,
                    traderObject,
                    option_position,
                    state.timestamp
                )
                result[Product.VOLCANIC_ROCK_VOUCHER_10500] = option_orders

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData