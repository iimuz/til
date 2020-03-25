# PostgreSQL Tips

## テーブルのデータ容量確認

- 2019.5.17 Qiita [PostgreSQL で各テーブルの総サイズと平均サイズを知る][awakia]

### 簡易バージョン

```sql
SELECT
  relname,
  reltuples,
  (relpages / 128) as mbytes,
  (relpages * 8192.0 / (reltuples + 1e-10)) as average_row_size
FROM pg_class
ORDER BY mbytes DESC;
```

### pg_toast などを考慮するバージョン

```sql
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
```

[awakia]: https://qiita.com/awakia/items/99c3d114aa16099e825d
