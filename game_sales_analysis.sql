select * from game_sales;

SELECT COUNT(*) AS total_games
FROM game_sales;

SELECT game, global
FROM game_sales
ORDER BY global DESC
LIMIT 10;

SELECT genre,
       ROUND(SUM(global),2) AS total_sales
FROM game_sales
GROUP BY genre
ORDER BY total_sales DESC;

SELECT platform,
       ROUND(SUM(global),2) AS global_sales
FROM game_sales
GROUP BY platform
ORDER BY global_sales DESC;

SELECT publisher,
       COUNT(*) AS total_games,
       ROUND(SUM(global),2) AS total_sales
FROM game_sales
GROUP BY publisher
ORDER BY total_sales DESC;

SELECT dominant_region,
       COUNT(*) AS games_count
FROM game_sales
GROUP BY dominant_region;

SELECT game, platform, europe
FROM game_sales
WHERE dominant_region = 'Europe'
ORDER BY europe DESC;

SELECT sales_tier,
       COUNT(*) AS games
FROM game_sales
GROUP BY sales_tier;

SELECT genre, platform,
       ROUND(AVG(global),2) AS avg_global_sales
FROM game_sales
WHERE sales_tier = 'Blockbuster'
GROUP BY genre, platform
ORDER BY avg_global_sales DESC;

SELECT game, genre, global,
       RANK() OVER (PARTITION BY genre ORDER BY global DESC) AS rank_in_genre
FROM game_sales;

SELECT year,
       ROUND(SUM(global),2) AS yearly_sales
FROM game_sales
GROUP BY year
ORDER BY year;