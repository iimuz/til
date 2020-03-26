/*
  選択した Database 中のテーブル毎の容量を出力します。
  Ref: https://qiita.com/awakia/items/99c3d114aa16099e825d
*/

SELECT
  relname,
  reltuples,
  (relpages / 128) as mbytes,
  (relpages * 8192.0 / (reltuples + 1e-10)) as average_row_size
FROM pg_class
ORDER BY mbytes DESC;
