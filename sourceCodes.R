library(forecast)
forecastStl <- function(x, n.ahead = 30) {
  myTs <- ts(x$PDS_CLOSE_RATE, start = 1, frequency = 256)
  fit.stl <- stl(myTs, s.window = 256)
  sts <- fit.stl$time.series
  trend <- sts[, "trend"]
  fore <- forecast(fit.stl, h = n.ahead, level = 95)
  plot(fore)
  pred <- fore$mean
  upper <- fore$upper
  lower <- fore$lower
  output <- data.frame(actual = c(x$PDS_CLOSE_RATE, rep(NA, n.ahead)), 
                       trend = c(trend, rep(NA, n.ahead)), pred = c(rep(NA, nrow(x)), pred), lower = c(rep(NA, nrow(x)), lower), 
                       upper = c(rep(NA, nrow(x)), upper), date = c(x$Date, max(x$Date) + (1:n.ahead)))
  return(output)
}

forecastArima <- function(x, n.ahead = 30) {
  myTs <- ts(x$PDS_CLOSE_RATE, start = 1, frequency = 366)
  fit.arima <- arima(myTs, order = c(0, 0, 1))
  fore <- forecast(fit.arima, h = n.ahead)
  plot(fore)
  upper <- fore$upper[, "95%"]
  lower <- fore$lower[, "95%"]
  trend <- as.numeric(fore$fitted)
  pred <- as.numeric(fore$mean)
  output <- data.frame(actual = c(x$PDS_CLOSE_RATE, rep(NA, n.ahead)), 
                       trend = c(trend, rep(NA, n.ahead)), pred = c(rep(NA, nrow(x)), pred), lower = c(rep(NA, nrow(x)), lower), 
                       upper = c(rep(NA, nrow(x)), upper), date = c(x$DATE, max(x$DATE) + (1:n.ahead)))
  return(output)
}


plotForecastResult <- function(x, title = NULL) {
  x <- x[order(x$DATE),]
  max.val <- max(c(x$actual, x$upper), na.rm = T)
  min.val <- min(c(x$actual, x$lower), na.rm = T)
  plot(x$DATE, x$actual, type = "l", col = "grey", main = title, 
       xlab = "Time", ylab = "Exchange Rate", xlim = range(x$DATE), ylim = c(min.val, max.val))
  grid()
  lines(x$DATE, x$trend, col = "yellowgreen")
  lines(x$DATE, x$pred, col = "green")
  lines(x$DATE, x$lower, col = "blue")
  lines(x$DATE, x$upper, col = "blue")
  legend("bottomleft", col = c("grey", "yellowgreen", "green", "blue"), lty = 1, c("Actual", "Trend", "Forecast", "Lower/Upper Bound"))
  }






