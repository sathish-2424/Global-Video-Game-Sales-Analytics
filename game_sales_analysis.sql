create database games;
use games;

show tables;
select * from games;

SELECT COUNT(*) AS total_rows
FROM games;

### null value find each Column
SELECT
    COUNT(*) - COUNT(Name) AS Name_nulls,
    COUNT(*) - COUNT(Year) AS Year_nulls,
    COUNT(*) - COUNT(Genre) AS Genre_nulls,
    COUNT(*) - COUNT(Publisher) AS Publisher_nulls,
    COUNT(*) - COUNT(Developer) AS Developer_nulls,
    COUNT(*) - COUNT(Platform) AS Platform_nulls,
    COUNT(*) - COUNT(Units) AS Units_nulls,
    COUNT(*) - COUNT(Revenue_USD_M) AS Revenue_nulls
FROM games;



### Total Revenue Generated
SELECT 
    SUM(Revenue_USD_M) AS total_revenue_million_usd
FROM games;

### Top 10 Best-Selling Games (by Units)
SELECT 
    Name,
    SUM(Units) AS total_units_sold
FROM games
GROUP BY Name
ORDER BY total_units_sold DESC
LIMIT 10;

### Revenue by Genre
SELECT 
    Genre,
    SUM(Revenue_USD_M) AS total_revenue
FROM games
GROUP BY Genre
ORDER BY total_revenue DESC;

### Top Publishers by Revenue
SELECT 
    Publisher,
    SUM(Revenue_USD_M) AS revenue
FROM games
GROUP BY Publisher
ORDER BY revenue DESC
LIMIT 5;

### Platform Performance (Units Sold)
SELECT 
    Platform,
    SUM(Units) AS total_units
FROM games
GROUP BY Platform
ORDER BY total_units DESC;

### Year-Wise Revenue Trend
SELECT 
    Year,
    SUM(Revenue_USD_M) AS yearly_revenue
FROM games
GROUP BY Year
ORDER BY Year;

### Most Profitable Game per Year
SELECT 
    Year,
    Name,
    SUM(Revenue_USD_M) AS revenue
FROM games
GROUP BY Year, Name
ORDER BY Year, revenue DESC;

### Average Revenue per Game by Genre
SELECT 
    Genre,
    AVG(Revenue_USD_M) AS avg_revenue
FROM games
GROUP BY Genre
ORDER BY avg_revenue DESC;

### Games Released on Multiple Platforms
SELECT 
    Name,
    COUNT(DISTINCT Platform) AS platform_count
FROM games
GROUP BY Name
HAVING COUNT(DISTINCT Platform) > 1
ORDER BY platform_count DESC;

### Top Developer by Total Revenue
SELECT 
    Developer,
    SUM(Revenue_USD_M) AS total_revenue
FROM games
GROUP BY Developer
ORDER BY total_revenue DESC
LIMIT 1;

### Rank Games by Revenue Within Each Year
SELECT 
    Year,
    Name,
    Revenue_USD_M,
    RANK() OVER (PARTITION BY Year ORDER BY Revenue_USD_M DESC) AS revenue_rank
FROM games;





