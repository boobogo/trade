#define Bid SymbolInfoDouble(_Symbol,SYMBOL_BID)
#define Ask SymbolInfoDouble(_Symbol,SYMBOL_ASK)
#include <Trade\Trade.mqh>
#include <Trade\TerminalInfo.mqh>
CTrade trade;
CTerminalInfo terminal;

bool IsTradingTime()
{
    // Get current time as datetime
    datetime current_time = TimeCurrent();
    
    // Convert to human-readable structure
    MqlDateTime time_struct;
    TimeToStruct(current_time, time_struct);

    // Extract hours and minutes
    int current_hour = time_struct.hour;

    // Define trading session hours in local time zone
    int start_hour = 12;
    int end_hour = 20;

    // Check if the current time is within the trading session
    if (current_hour >= start_hour && current_hour < end_hour)
        return true; // It's within trading hours
    else
        return false; // Outside trading hours
}


bool IsNewCandle()
  {
    static datetime prevTime=0;
    datetime lastTime[1];
    if (CopyTime(_Symbol,_Period,0,1,lastTime)==1 && prevTime!=lastTime[0])
    {
         prevTime=lastTime[0];
         return true;
    }
   return false;
  }


double tp_buy_macd_prev_cross(string symbol, ENUM_TIMEFRAMES timeframe)
{
    int macd_period_fast = 12;    // Fast EMA period
    int macd_period_slow = 26;   // Slow EMA period
    int macd_signal_period = 9;  // Signal SMA period
    int bars_to_check = 1000;    // Number of historical bars to check

    // Buffers for MACD indicator
    double macd_line[];
    double signal_line[];
    
    // Get handle for MACD
    int macd_handle = iMACD(NULL, timeframe, macd_period_fast, macd_period_slow, macd_signal_period, PRICE_CLOSE);

    if (macd_handle == INVALID_HANDLE)
    {
        Print("Failed to create MACD handle.");
        return 0.0;
    }

    // Resize buffers
    ArraySetAsSeries(macd_line, true);
    ArraySetAsSeries(signal_line, true);

    // Copy MACD data
    if (CopyBuffer(macd_handle, 0, 0, bars_to_check, macd_line) <= 0 ||
        CopyBuffer(macd_handle, 1, 0, bars_to_check, signal_line) <= 0)
    {
        Print("Failed to copy MACD data.");
        IndicatorRelease(macd_handle);
        return 0.0;
    }

    // Release the MACD handle
    IndicatorRelease(macd_handle);

    // Get current price
    double current_price = SymbolInfoDouble(symbol, SYMBOL_BID);

    // Loop through historical bars to find the cross-down level
    for (int i = 1; i < 200; i++)  // Start from 1 (0 is the current bar)
    {
        // Check if MACD line crossed below the signal line
        if (macd_line[i] < signal_line[i] && macd_line[i + 1] > signal_line[i + 1])
        {
            // Get the price level of the cross-down
            double crossdown_price = iLow(symbol, timeframe, i);

            // Check if the cross-down price is higher than the current price
            if (crossdown_price > current_price)
                return crossdown_price;  // Return the TP level
        }
    }

    // No valid TP level found
    Print("No suitable TP level found.");
    return 0.0;
}


double sl_buy_macd_prev_cross(string symbol, ENUM_TIMEFRAMES timeframe)
   {
   int macd_handle = iMACD(NULL,timeframe,12,26,9,PRICE_CLOSE);
   double macdBuffer[];
   double signalBuffer[];
   ArraySetAsSeries(macdBuffer,true);
   ArraySetAsSeries(signalBuffer,true);
   CopyBuffer(macd_handle,0,0,200,macdBuffer);
   CopyBuffer(macd_handle,1,0,200,signalBuffer);
   int i = 0;
   double sl = 0.0;
   
   // Release the MACD handle
    IndicatorRelease(macd_handle);
   
   double minStopLevel = SymbolInfoInteger(Symbol(),SYMBOL_TRADE_STOPS_LEVEL)*Point();
   while(i<200)
     {
      if(macdBuffer[i]<signalBuffer[i])
      {
        sl = iLow(NULL,timeframe,i);
        
        if(sl<Bid)
          return(Bid-sl >= minStopLevel ? sl : Bid-minStopLevel);
      }
      i++;
     }
   return sl;
   }


double tp_buy_stoch_prev_cross(string symbol, ENUM_TIMEFRAMES timeframe)
{
    int k_period = 14;         // %K period
    int d_period = 3;          // %D period
    int slowing = 3;           // Slowing
    int bars_to_check = 1000;  // Number of historical bars to check

    // Buffers for Stochastic Oscillator
    double k_line[];
    double d_line[];

    // Get handle for Stochastic Oscillator
    int stochastic_handle = iStochastic(symbol, timeframe, k_period, d_period, slowing, MODE_SMA, STO_LOWHIGH);

    if (stochastic_handle == INVALID_HANDLE)
    {
        Print("Failed to create Stochastic handle.");
        return 0.0;
    }

    // Resize buffers
    ArraySetAsSeries(k_line, true);
    ArraySetAsSeries(d_line, true);

    // Copy Stochastic data
    if (CopyBuffer(stochastic_handle, 0, 0, bars_to_check, k_line) <= 0 ||  // %K line
        CopyBuffer(stochastic_handle, 1, 0, bars_to_check, d_line) <= 0)    // %D line
    {
        Print("Failed to copy Stochastic data.");
        IndicatorRelease(stochastic_handle);
        return 0.0;
    }

    // Release the Stochastic handle
    IndicatorRelease(stochastic_handle);

    // Get current price
    double current_price = SymbolInfoDouble(symbol, SYMBOL_BID);

    // Loop through historical bars to find the cross-down level
    for (int i = 1; i < 200; i++)  // Start from 1 (0 is the current bar)
    {
        // Check if %K line crossed below the %D line
        if (k_line[i] < d_line[i] && k_line[i + 1] > d_line[i + 1])
        {
            // Get the price level of the cross-down
            double crossdown_price = iLow(symbol, timeframe, i);

            // Check if the cross-down price is higher than the current price
            if (crossdown_price > current_price)
                return crossdown_price;  // Return the TP level
        }
    }

    // No valid TP level found
    Print("No suitable TP level found.");
    return 0.0;
}


void TrailingStopLoss()
{
   ulong ticket = PositionGetTicket(0); // Get the ticket of the first position
   if(ticket == INVALID_HANDLE)
      return; // No position found

   // Get trade details
   double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
   double current_price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   double tp = PositionGetDouble(POSITION_TP);
   double sl = PositionGetDouble(POSITION_SL);

   // Ensure TP and SL are valid
   if(tp <= 0 || sl <= 0)
      return;

   // Calculate the 50% level to TP
   double halfway_to_tp = entry_price + (tp - entry_price) * 0.5;

   // Move the stop loss to breakeven when price reaches halfway to TP
   if(current_price >= halfway_to_tp && sl < entry_price) // Only modify if SL is below breakeven
   {
      double breakeven_level = NormalizeDouble(entry_price, Digits());
      if(!trade.PositionModify(ticket, breakeven_level, tp))
         Print("Failed to move stop loss to breakeven. Error: ", GetLastError());
      else
         Print("Stop loss moved to breakeven at ", breakeven_level);
   }
}

bool check_trade_conditions(int method, bool use_macd1, bool use_macd2, bool use_macd3, ENUM_TIMEFRAMES tf1, ENUM_TIMEFRAMES tf2, ENUM_TIMEFRAMES tf3, ENUM_TIMEFRAMES trigger_tf, int trigger_method)
{
   bool condition1 = true, condition2 = true, condition3 = true;

   if (use_macd1)
      condition1 = check_macd_filter_condition(tf1);

   if (use_macd2)
      condition2 = check_macd_filter_condition(tf2);
      
   if (use_macd3)
      condition3 = check_macd_filter_condition(tf3);
      
   // Check trigger condition
   bool trigger_condition = check_trigger_condition(trigger_tf, trigger_method);

   return (condition1 && condition2 && condition3 && trigger_condition);
}


bool check_macd_filter_condition(ENUM_TIMEFRAMES timeframe)
{
   double macdBuffer[], signalBuffer[];
   int macdHandle = iMACD(NULL, timeframe, 12, 26, 9, PRICE_CLOSE);

   ArraySetAsSeries(macdBuffer, true);
   ArraySetAsSeries(signalBuffer, true);

   if (CopyBuffer(macdHandle, 0, 0, 1, macdBuffer) < 1 ||
       CopyBuffer(macdHandle, 1, 0, 1, signalBuffer) < 1)
      return false;

   return macdBuffer[0] > signalBuffer[0];
}


bool check_trigger_condition(ENUM_TIMEFRAMES timeframe, int trigger_method)
{
    if (trigger_method == 0) // Use MACD cross-up
        return check_macd_crossup(timeframe);
    else if (trigger_method == 1) // Use Stochastic cross-up
        return check_stochastic_crossup(timeframe);
    
    return false; // Default: No condition met
}


bool check_macd_crossup(ENUM_TIMEFRAMES timeframe)
{
    int macd_handle = iMACD(NULL, timeframe, 12, 26, 9, PRICE_CLOSE); // MACD parameters: 12, 26, 9
    double macd_buffer[], signal_buffer[];
    
    // Set arrays as series
    ArraySetAsSeries(macd_buffer, true);
    ArraySetAsSeries(signal_buffer, true);

    // Copy MACD line and Signal line data
    if (CopyBuffer(macd_handle, 0, 0, 3, macd_buffer) <= 0 || CopyBuffer(macd_handle, 1, 0, 3, signal_buffer) <= 0)
    {
        Print("Error copying MACD data");
        return false; // If there's an error retrieving data, return false
    }

    // Check for cross-up: MACD[1] < Signal[1] and MACD[0] > Signal[0]
    if (macd_buffer[2] < signal_buffer[2] && macd_buffer[1] > signal_buffer[1])
        return true;

    return false;
}


bool check_stochastic_crossup(ENUM_TIMEFRAMES timeframe)
{
    int stochastic_handle = iStochastic(NULL, timeframe, 5, 3, 3, MODE_SMA, STO_LOWHIGH); // Stochastic parameters: 5, 3, 3
    double k_buffer[], d_buffer[];

    // Set arrays as series
    ArraySetAsSeries(k_buffer, true);
    ArraySetAsSeries(d_buffer, true);

    // Copy %K and %D line data
    if (CopyBuffer(stochastic_handle, 0, 0, 3, k_buffer) <= 0 || CopyBuffer(stochastic_handle, 1, 0, 3, d_buffer) <= 0)
    {
        Print("Error copying Stochastic data");
        return false; // If there's an error retrieving data, return false
    }

    // Check for cross-up: %K[1] < %D[1] and %K[0] > %D[0]
    if (k_buffer[2] < d_buffer[2] && k_buffer[1] > d_buffer[1])
        return true;

    return false;
}


enum ENUM_STRATEGY
{
   STRATEGY_A,  // 0
   STRATEGY_B,  // 1
   STRATEGY_C   // 2
};