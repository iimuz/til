"""MT4の履歴を管理するモジュール."""
import sqlalchemy as sa

from .db import Base


class History(Base):
    """MT4の履歴情報を扱う."""

    __tablename__ = "history"

    ticket = sa.Column(sa.Integer(), primary_key=True, nullable=False)  # チケット番号
    open_time = sa.Column(sa.DateTime(timezone=True), nullable=False)  # 注文時間
    open_price = sa.Column(sa.Float(8), nullable=False)  # 注文価格
    type = sa.Column(sa.Integer(), nullable=False)  # 注文タイプ
    lots = sa.Column(sa.Float(8), nullable=False)  # ロット数
    symbol = sa.Column(sa.String(20), nullable=False)  # 通貨ペア名称
    stop_loss = sa.Column(sa.Float(8), nullable=False)  # ストップロス価格
    take_profit = sa.Column(sa.Float(8), nullable=False)  # 損益
    close_time = sa.Column(sa.DateTime(timezone=True), nullable=False)  # 決済時間
    close_price = sa.Column(sa.Float(8), nullable=False)  # 決済価格
    commission = sa.Column(sa.Float(8), nullable=False)  # 手数料
    expiration = sa.Column(sa.DateTime(timezone=True), nullable=False)  # 有効期限
    swap = sa.Column(sa.Float(8), nullable=False)  # スワップ損益
    profit = sa.Column(sa.Float(8), nullable=False)  # リミット価格
    comment = sa.Column(sa.String(256), nullable=False)  # コメント
    magic_number = sa.Column(sa.Integer(), nullable=False)  # 識別番号
