#include <strategy_header.mqh>

input int loss_week = 8; //in percetange
input int loss_day =  3;
input double lot = 0.01;
input double rr_min = 3;

input ENUM_TIMEFRAMES tf1 = PERIOD_M30;        // Timeframe for first filter
input ENUM_TIMEFRAMES tf2 = PERIOD_H1;        // Timeframe for second filter
input ENUM_TIMEFRAMES tf3 = PERIOD_H4;       // Timeframe for third filter
input ENUM_TIMEFRAMES trigger_tf = PERIOD_M15;   // Timeframe for trigger
input ENUM_TIMEFRAMES sl_tf = PERIOD_M5;        // Timeframe for sl
input ENUM_TIMEFRAMES tp_tf = PERIOD_H4;        //Timeframe for tp

input bool use_macd1 = true;                  // Use MACD on tf1
input bool use_macd2 = true;                  // Use MACD on tf2
input bool use_macd3 = true;                  // Use MACD on tf3

input int trigger_method = 0;            // trigger method 0-macd,1-stoch
input int tp_method = 0;                 // tp method 0-macd_prev,1-macd_next, 2-stoch_prev, 3-stoch_next
input int sl_method = 0;                 // sl method 0-macd_prev,1-stoch_prev


input int trailing_sl_breakeven= 30;      //trailing sl to breakeven when 30% of tp reached
input ENUM_STRATEGY STRATEGY_A;


void OnTick()
{   
   // Check if it's within the trading session
   if (!IsTradingTime()) return;
        
   if(IsNewCandle())
   {                                                 
     if(PositionsTotal()==0)        
        {
          if(check_trade_conditions(trigger_method, use_macd1, use_macd2, use_macd3, tf1, tf2, tf3,trigger_tf,trigger_method))
          { 
            
            double sl = NormalizeDouble(sl_buy_macd_prev_cross(Symbol(),sl_tf),Digits());
            double tp = NormalizeDouble(tp_buy_macd_prev_cross(Symbol(),tp_tf),Digits());
            
            if(tp!=0 && sl!=0 && (tp-Ask)/(Ask-sl)> rr_min)
              trade.Buy(lot,NULL,Ask,sl,tp,NULL);
          }
        }
      
     Comment(
              "Equity : ", AccountInfoDouble(ACCOUNT_EQUITY),"\n",
              "trade time: ",IsTradingTime(), "\n"
              ,SymbolInfoInteger(Symbol(),SYMBOL_TRADE_STOPS_LEVEL)*Point()); 
      }

}

