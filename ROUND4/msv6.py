

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math


class Trader:
    """
    IMC Prosperity Challenge Trader
    Multi-strategy algorithmic trading system for Round 4
    
    Format:
        - run(state: TradingState) -> Tuple[Dict, int, str]
        - bid() -> int (for Round 2 compatibility)
    """
    
    def __init__(self):
        """Initialize trader state and parameters"""
        
        # -------- PRODUCT DEFINITIONS --------
        self.HYDROGEL = "HYDROGEL_PACK"
        self.VELVETFRUIT = "VELVETFRUIT_EXTRACT"
        self.VEV_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
        
        # -------- POSITION TRACKING --------
        self.positions: Dict[str, int] = defaultdict(int)
        self.position_limits: Dict[str, int] = {
            self.HYDROGEL: 200,
            self.VELVETFRUIT: 200,
        }
        for strike in self.VEV_STRIKES:
            self.position_limits[f"VEV_{strike}"] = 300
        
        # -------- COUNTERPARTY INTELLIGENCE --------
        self.counterparty_net_position: Dict[UserId, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.counterparty_trade_count: Dict[UserId, int] = defaultdict(int)
        
        # Known aggressive traders
        self.BUYERS = {'Mark 01', 'Mark 67'}
        self.SELLERS = {'Mark 22'}
        self.MM_TRADERS = {'Mark 14', 'Mark 38', 'Mark 55'}
        
        # -------- MARKET STATISTICS --------
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        self.max_history_length = 300
        self.mean_prices: Dict[str, float] = {}
        self.std_devs: Dict[str, float] = {}
        
        # -------- MOMENTUM TRACKING --------
        self.momentum_samples: Dict[str, List[float]] = defaultdict(list)
        self.momentum_window = 20
        
        # -------- STRATEGY STATE --------
        self.last_order_time: Dict[str, int] = defaultdict(int)
        self.min_order_interval = 50
        self.last_pair_trade_time = 0
        self.min_pair_interval = 100
        
        # -------- RISK MANAGEMENT --------
        self.max_single_order = 30
        self.max_position_size = 150
        
        # -------- LOGGING --------
        self.action_log: List[str] = []

    def bid(self) -> int:
        """Required method for Round 2 compatibility. Ignored in other rounds."""
        return 15

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Main algorithm entry point.
        Challenge format: returns (result_dict, conversions, traderData)
        
        Args:
            state: TradingState with order depths and trade history
            
        Returns:
            Tuple of:
            - result: Dict[product_name] -> List[Order]
            - conversions: int (0 for no conversions)
            - traderData: str (state preservation for next call)
        """
        
        orders_dict = {}
        
        try:
            # 1. UPDATE MARKET INTELLIGENCE
            self._update_counterparty_intelligence(state)
            self._update_market_statistics(state)
            self._update_momentum(state)
            
            # 2. GENERATE ORDERS FROM EACH STRATEGY LAYER
            orders_dict.update(self._counterparty_following(state))
            orders_dict.update(self._pair_trading(state))
            orders_dict.update(self._momentum_trading(state))
            orders_dict.update(self._mean_reversion(state))
            orders_dict.update(self._risk_management(state))
            
            # 3. ENSURE ALL PRODUCTS HAVE ORDER LISTS
            for product in state.order_depths.keys():
                if product not in orders_dict:
                    orders_dict[product] = []
            
            # 4. FINAL VALIDATION
            orders_dict = self._validate_orders(orders_dict, state)
            
        except Exception as e:
            # Safety: return empty orders on error
            self._log(f"ERROR: {str(e)}")
            for product in state.order_depths.keys():
                orders_dict[product] = []
        
        # No conversions for Round 4
        conversions = 0
        traderData = ""
        
        # Challenge format: return tuple
        return orders_dict, conversions, traderData

    # ======================== STRATEGY LAYERS ========================
    
    def _counterparty_following(self, state: TradingState) -> Dict[str, List[Order]]:
        """LAYER 1: Follow aggressive traders (Mark 01, Mark 67, Mark 22)"""
        orders = defaultdict(list)
        
        try:
            buyer_positions = {}
            for buyer in self.BUYERS:
                buyer_positions[buyer] = self.counterparty_net_position[buyer]
            
            seller_positions = {}
            for seller in self.SELLERS:
                seller_positions[seller] = self.counterparty_net_position[seller]
            
            for symbol in [self.HYDROGEL, self.VELVETFRUIT]:
                mid = self._get_mid_price(state, symbol)
                if mid is None:
                    continue
                
                buyer_momentum = sum(pos.get(symbol, 0) for pos in buyer_positions.values())
                seller_momentum = sum(pos.get(symbol, 0) for pos in seller_positions.values())
                
                # BUY signal
                if buyer_momentum > 30 and self._can_trade(symbol, 20):
                    size = min(20, self.position_limits[symbol] - self.positions[symbol])
                    if size > 0:
                        orders[symbol].append(Order(symbol, int(mid - 1), size))
                
                # SELL signal
                if seller_momentum < -30 and self._can_trade(symbol, -20):
                    size = min(20, self.positions[symbol])
                    if size > 0:
                        orders[symbol].append(Order(symbol, int(mid + 1), -size))
        
        except Exception as e:
            self._log(f"ERROR in counterparty_following: {str(e)}")
        
        return orders
    
    def _pair_trading(self, state: TradingState) -> Dict[str, List[Order]]:
        """LAYER 2: Pair trading between HYDROGEL and VELVETFRUIT"""
        orders = defaultdict(list)
        
        if state.timestamp - self.last_pair_trade_time < self.min_pair_interval:
            return orders
        
        try:
            hydro_mid = self._get_mid_price(state, self.HYDROGEL)
            vext_mid = self._get_mid_price(state, self.VELVETFRUIT)
            
            if hydro_mid is None or vext_mid is None:
                return orders
            
            hydro_spread = self._get_spread(state, self.HYDROGEL)
            vext_spread = self._get_spread(state, self.VELVETFRUIT)
            
            if hydro_spread > vext_spread * 1.8 and self._can_trade(self.VELVETFRUIT, 15) \
                    and self._can_trade(self.HYDROGEL, -15):
                orders[self.VELVETFRUIT].append(Order(self.VELVETFRUIT, int(vext_mid - 1), 15))
                orders[self.HYDROGEL].append(Order(self.HYDROGEL, int(hydro_mid + 1), -15))
                self.last_pair_trade_time = state.timestamp
            
            elif vext_spread > hydro_spread * 1.5 and self._can_trade(self.HYDROGEL, 15) \
                    and self._can_trade(self.VELVETFRUIT, -15):
                orders[self.HYDROGEL].append(Order(self.HYDROGEL, int(hydro_mid - 1), 15))
                orders[self.VELVETFRUIT].append(Order(self.VELVETFRUIT, int(vext_mid + 1), -15))
                self.last_pair_trade_time = state.timestamp
        
        except Exception as e:
            self._log(f"ERROR in pair_trading: {str(e)}")
        
        return orders
    
    def _momentum_trading(self, state: TradingState) -> Dict[str, List[Order]]:
        """LAYER 3: Momentum trading on ATM VEV products"""
        orders = defaultdict(list)
        
        try:
            atm_strikes = [5100, 5200, 5300]
            
            for strike in atm_strikes:
                symbol = f"VEV_{strike}"
                if symbol not in state.order_depths:
                    continue
                
                mid = self._get_mid_price(state, symbol)
                if mid is None:
                    continue
                
                momentum = self._calculate_momentum(symbol)
                
                if momentum > 0.6 and self._can_trade(symbol, 10):
                    spread = self._get_spread(state, symbol)
                    if spread < 5:
                        order_depth = state.order_depths.get(symbol, OrderDepth())
                        if order_depth.sell_orders:
                            ask_price = min(order_depth.sell_orders.keys())
                            size = min(10, abs(order_depth.sell_orders[ask_price]))
                            if size > 0 and self._can_trade(symbol, size):
                                orders[symbol].append(Order(symbol, int(ask_price), size))
                
                elif momentum < -0.6 and self._can_trade(symbol, -10):
                    spread = self._get_spread(state, symbol)
                    if spread < 5:
                        order_depth = state.order_depths.get(symbol, OrderDepth())
                        if order_depth.buy_orders:
                            bid_price = max(order_depth.buy_orders.keys())
                            size = min(10, order_depth.buy_orders[bid_price])
                            if size > 0 and self._can_trade(symbol, -size):
                                orders[symbol].append(Order(symbol, int(bid_price), -size))
        
        except Exception as e:
            self._log(f"ERROR in momentum_trading: {str(e)}")
        
        return orders
    
    def _mean_reversion(self, state: TradingState) -> Dict[str, List[Order]]:
        """LAYER 4: Mean reversion on core products"""
        orders = defaultdict(list)
        
        try:
            for symbol in [self.HYDROGEL, self.VELVETFRUIT]:
                if symbol not in state.order_depths:
                    continue
                
                mid = self._get_mid_price(state, symbol)
                
                if symbol not in self.mean_prices or symbol not in self.std_devs:
                    continue
                
                mean = self.mean_prices[symbol]
                std = self.std_devs[symbol]
                
                # Sell rallies
                if mid > mean + 1.5 * std:
                    if self._can_trade(symbol, -10):
                        size = min(10, self.positions[symbol])
                        if size > 0:
                            orders[symbol].append(Order(symbol, int(mid), -size))
                
                # Buy dips
                elif mid < mean - 1.5 * std:
                    if self._can_trade(symbol, 10):
                        size = min(10, self.position_limits[symbol] - self.positions[symbol])
                        if size > 0:
                            orders[symbol].append(Order(symbol, int(mid), size))
        
        except Exception as e:
            self._log(f"ERROR in mean_reversion: {str(e)}")
        
        return orders
    
    def _risk_management(self, state: TradingState) -> Dict[str, List[Order]]:
        """LAYER 5: Position limit enforcement and rebalancing"""
        orders = defaultdict(list)
        
        try:
            for symbol in list(self.positions.keys()):
                if symbol not in state.order_depths:
                    continue
                
                position = self.positions[symbol]
                limit = self.position_limits.get(symbol, 200)
                
                mid = self._get_mid_price(state, symbol)
                if mid is None:
                    continue
                
                # Reduce overweight long positions
                if position > self.max_position_size:
                    excess = position - self.max_position_size
                    orders[symbol].append(Order(symbol, int(mid + 1), -excess))
                
                # Reduce overweight short positions
                elif position < -self.max_position_size:
                    excess = -position - self.max_position_size
                    orders[symbol].append(Order(symbol, int(mid - 1), excess))
        
        except Exception as e:
            self._log(f"ERROR in risk_management: {str(e)}")
        
        return orders

    # ======================== HELPER FUNCTIONS ========================
    
    def _update_counterparty_intelligence(self, state: TradingState):
        """Track counterparty positions from market trades"""
        try:
            for symbol, trades in state.market_trades.items():
                for trade in trades:
                    if trade.buyer:
                        self.counterparty_net_position[trade.buyer][symbol] += trade.quantity
                    if trade.seller:
                        self.counterparty_net_position[trade.seller][symbol] -= trade.quantity
        except Exception as e:
            self._log(f"ERROR updating counterparty: {str(e)}")
    
    def _update_market_statistics(self, state: TradingState):
        """Update rolling price statistics"""
        try:
            for symbol in [self.HYDROGEL, self.VELVETFRUIT]:
                mid = self._get_mid_price(state, symbol)
                
                if mid is not None:
                    self.price_history[symbol].append(mid)
                    if len(self.price_history[symbol]) > self.max_history_length:
                        self.price_history[symbol].pop(0)
            
            for symbol in [self.HYDROGEL, self.VELVETFRUIT]:
                if len(self.price_history[symbol]) > 10:
                    prices = self.price_history[symbol]
                    mean = sum(prices) / len(prices)
                    self.mean_prices[symbol] = mean
                    
                    variance = sum((p - mean) ** 2 for p in prices) / len(prices)
                    self.std_devs[symbol] = math.sqrt(variance) if variance > 0 else 0.1
        
        except Exception as e:
            self._log(f"ERROR updating statistics: {str(e)}")
    
    def _update_momentum(self, state: TradingState):
        """Update momentum tracking"""
        try:
            symbols_to_track = [self.HYDROGEL, self.VELVETFRUIT] + \
                              [f"VEV_{s}" for s in [5100, 5200, 5300]]
            
            for symbol in symbols_to_track:
                mid = self._get_mid_price(state, symbol)
                
                if mid is not None:
                    self.momentum_samples[symbol].append(mid)
                    if len(self.momentum_samples[symbol]) > self.momentum_window:
                        self.momentum_samples[symbol].pop(0)
        
        except Exception as e:
            self._log(f"ERROR updating momentum: {str(e)}")
    
    def _calculate_momentum(self, symbol: str) -> float:
        """Calculate momentum score (-1 to +1)"""
        try:
            if symbol not in self.momentum_samples:
                return 0.0
            
            samples = self.momentum_samples[symbol]
            
            if len(samples) < 5:
                return 0.0
            
            recent_avg = sum(samples[-5:]) / 5
            older_avg = sum(samples[:5]) / 5
            
            if older_avg == 0:
                return 0.0
            
            momentum = (recent_avg - older_avg) / older_avg
            return max(-1.0, min(1.0, momentum * 10))
        
        except Exception:
            return 0.0
    
    def _get_mid_price(self, state: TradingState, symbol: str) -> Optional[float]:
        """Get mid price from order book"""
        try:
            order_depth = state.order_depths.get(symbol)
            if order_depth is None:
                return None
            
            if not order_depth.buy_orders or not order_depth.sell_orders:
                return None
            
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            
            return (best_bid + best_ask) / 2.0
        
        except Exception:
            return None
    
    def _get_spread(self, state: TradingState, symbol: str) -> float:
        """Get bid-ask spread"""
        try:
            order_depth = state.order_depths.get(symbol)
            if order_depth is None:
                return 999
            
            if not order_depth.buy_orders or not order_depth.sell_orders:
                return 999
            
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            
            return best_ask - best_bid
        
        except Exception:
            return 999
    
    def _can_trade(self, symbol: str, delta: int) -> bool:
        """Check if trade is allowed (position limits)"""
        new_position = self.positions[symbol] + delta
        limit = self.position_limits.get(symbol, 200)
        
        return -limit <= new_position <= limit
    
    def _validate_orders(self, orders_dict: Dict[str, List[Order]], 
                        state: TradingState) -> Dict[str, List[Order]]:
        """Validate all orders before submission"""
        validated = {}
        
        valid_symbols = set(state.order_depths.keys())
        
        for product in valid_symbols:
            validated[product] = []
            
            if product not in orders_dict:
                continue
            
            for order in orders_dict[product]:
                # Check price and quantity
                if order.price <= 0 or order.quantity == 0:
                    continue
                
                # Check position limits
                new_position = self.positions[product] + order.quantity
                limit = self.position_limits.get(product, 200)
                
                if -limit <= new_position <= limit:
                    validated[product].append(order)
                    self.positions[product] = new_position
        
        return validated
    
    def _log(self, message: str):
        """Log action message"""
        self.action_log.append(message)
        if len(self.action_log) > 100:
            self.action_log.pop(0)