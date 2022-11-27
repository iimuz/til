#property strict

#include <stdlib.mqh>

// 1件分の取引履歴を管理するためのクラス。
class OrderHistoryItem {
 private:
  int _ticket;           // チケット価格
  datetime _openTime;    // 注文時間
  double _openPrice;     // 注文価格
  int _type;             // 注文タイプ
  double _lots;          // ロット数
  string _symbol;        // 通貨ペアの名称
  double _stopLoss;      // ストップロス価格
  double _takeProfit;    // 損益
  datetime _closeTime;   // 決済時間
  double _closePrice;    // 決済価格
  double _commission;    // 手数料
  datetime _expiration;  // 有効期限
  double _swap;          // スワップ損益
  double _profit;        // リミット価格
  string _comment;       // コメント
  int _magicNumber;      // 識別番号

 public:
  OrderHistoryItem() {}

  // 指定した注文番号の情報を取得して設定する
  OrderHistoryItem(const int orderIndex) {
    const bool isOrderSelect =
        OrderSelect(orderIndex, SELECT_BY_POS, MODE_HISTORY);
    if (isOrderSelect == false) {
      const int error_code = GetLastError();
      printf("Select order error: error code(%d), detail: %s", error_code,
             ErrorDescription(error_code));
      return;
    }

    _ticket = OrderTicket();
    _openTime = OrderOpenTime();
    _openPrice = OrderOpenPrice();
    _type = OrderType();
    _lots = OrderLots();
    _symbol = OrderSymbol();
    _stopLoss = OrderStopLoss();
    _takeProfit = OrderTakeProfit();
    _closeTime = OrderCloseTime();
    _closePrice = OrderClosePrice();
    _commission = OrderCommission();
    _expiration = OrderExpiration();
    _swap = OrderSwap();
    _profit = OrderProfit();
    _comment = OrderComment();
    _magicNumber = OrderMagicNumber();

    return;
  }

  int ticket() const { return _ticket; }
  datetime openTime() const { return _openTime; }
  double openPrice() const { return _openPrice; }
  int type() const { return _type; }
  double lots() const { return _lots; }
  string symbol() const { return _symbol; }
  double stopLoss() const { return _stopLoss; }
  double takeProfit() const { return _takeProfit; }
  datetime closeTime() const { return _closeTime; }
  double closePrice() const { return _closePrice; }
  double commission() const { return _commission; }
  datetime expiration() const { return _expiration; }
  double swap() const { return _swap; }
  double profit() const { return _profit; }
  string comment() const { return _comment; }
  int magicNumber() const { return _magicNumber; }
};

// スクリプト実行のエントリポイント。
void OnStart() {
  const int fp =
      FileOpen("OrderHistory.csv", FILE_CSV | FILE_ANSI | FILE_WRITE, ',');
  WriteCSVHeader(fp);

  // MT4の口座履歴に表示されている期間に依存した要素数を取得
  const int TOTAL_ORDER = OrdersHistoryTotal();
  for (int orderIndex = 0; orderIndex < TOTAL_ORDER; ++orderIndex) {
    const OrderHistoryItem ORDER_ITEM(orderIndex);
    WriteCSVRow(fp, ORDER_ITEM);
  }

  FileClose(fp);
  Alert("Complete!");
}

// csvのヘッダ情報を記載する。
void WriteCSVHeader(const int fp) {
  FileWrite(fp, "ticket", "openTime", "openPrice", "type", "lots", "symbol",
            "stopLoss", "takeProfit", "closeTime", "closePrice", "commission",
            "expiration", "swap", "profit", "comment", "magicNumber");
}

// csvに1行分の取引履歴情報を記載する。
void WriteCSVRow(const int fp, const OrderHistoryItem& item) {
  FileWrite(fp, item.ticket(), item.openTime(), item.openPrice(), item.type(),
            item.lots(), item.symbol(), item.stopLoss(), item.takeProfit(),
            item.closeTime(), item.closePrice(), item.commission(),
            item.expiration(), item.swap(), item.profit(), item.comment(),
            item.magicNumber());
}
