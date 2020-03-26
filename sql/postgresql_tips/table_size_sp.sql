/*
  pgtoast などを考慮してテーブルサイズを表示します。
  Ref: https://qiita.com/awakia/items/99c3d114aa16099e825d
*/

SELECT
  pgn.nspname,
  relname,
  pg_size_pretty(relpages::bigint * 8 * 1024) AS size,
  CASE WHEN relkind = 't' THEN (
    SELECT pgd.relname
    FROM pg_class pgd
    WHERE pgd.reltoastrelid = pg.oid
  )
  WHEN
    nspname = 'pg_toast'
    AND relkind = 'i'
  THEN (
    SELECT pgt.relname
    FROM pg_class pgt
    WHERE
      SUBSTRING(pgt.relname FROM 10) = REPLACE(SUBSTRING(pg.relname FROM 10), '_index', '')
  ) ELSE (
    SELECT pgc.relname
    FROM pg_class pgc
    WHERE pg.reltoastrelid = pgc.oid
  ) END::varchar AS refrelname,
  CASE WHEN nspname = 'pg_toast' AND relkind = 'i' THEN (
    SELECT pgts.relname
    FROM pg_class pgts
    WHERE pgts.reltoastrelid = (
      SELECT pgt.oid
      FROM pg_class pgt
      WHERE SUBSTRING(pgt.relname FROM 10) = REPLACE(SUBSTRING(pg.relname FROM 10), '_index', '')
    )
  ) END AS relidxrefrelname,
  relfilenode,
  relkind,
  reltuples::bigint,
  relpages
FROM
  pg_class pg,
  pg_namespace pgn
WHERE
  pg.relnamespace = pgn.oid
  AND pgn.nspname NOT IN ('information_schema', 'pg_catalog')
ORDER BY relpages DESC;
