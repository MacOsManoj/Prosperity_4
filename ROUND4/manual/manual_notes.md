
  Trying to answer as many questions as possible in one go:                                                                                                                               
                                                                                                                                                                                          
  1) The goal is to maximise your PnL on the products you trade. We mark each product to the 'fair value' which is the average value of the product on 100 simulations                    
  2)  We are using solvinarian days and trading days throughout the wiki. Solvinarian days refer to days (in the same way we refer to days as 365 days a year) while trading days refer   
  to days where actual trading takes place (exchanges are open about 252 days a year)                                                                                                     
  3) No positions are taken to other rounds, we haven't processed any days yet. This is a standalone challenge.                                                                           
  4) The price columns doesnt represent the value it should for some. Its there only for cosmetic reasons and should show the 'investment cost'. You can ignore this column if you don't  
  find it sensible. It should not affect your decision in any way                                                                                                                         
  5) There is no buying/selling across days. You decide at t=0 if you buy or sell and what quantity. You hold the positions until expiry, at which they are market against their fair     
  value (see 1)

  moderators said this: TRADING_DAYS_PER_YEAR = 252                                                                                                     
  STEPS_PER_DAY = 4                                                                                                                                                                       
  STEPS_PER_YEAR = TRADING_DAYS_PER_YEAR * STEPS_PER_DAY                                                                                                                                  
                                                                                                                                                                                          
  def weeks_to_years(weeks: float) -> float:                                                                                                                                              
      # 5 business days per week, annualized to 252 trading days                                                                                                                          
      return (weeks * 5) / TRADING_DAYS_PER_YEAR                                                                                                                                          
                                                                                                                                                                                          
  def steps_for_weeks(weeks: float) -> int:                                                                                                                                               
      return int(round(weeks * 5 * STEPS_PER_DAY))                                                                                                                                                                                    
                  