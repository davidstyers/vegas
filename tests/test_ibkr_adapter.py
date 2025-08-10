import os
import pytest

from vegas.broker.ibkr_adapter import InteractiveBrokersBrokerAdapter, IBKRConfig
from vegas.strategy import Signal


def test_ibkr_config_from_env(monkeypatch):
    monkeypatch.setenv("IB_HOST", "10.0.0.2")
    monkeypatch.setenv("IB_PORT", "4002")
    monkeypatch.setenv("IB_CLIENT_ID", "42")
    monkeypatch.setenv("IB_ACCOUNT_ID", "DU1234567")
    cfg = IBKRConfig.from_env()
    assert cfg.host == "10.0.0.2"
    assert cfg.port == 4002
    assert cfg.client_id == 42
    assert cfg.account_id == "DU1234567"


def test_ibkr_offline_place_and_cancel():
    broker = InteractiveBrokersBrokerAdapter(IBKRConfig(host="127.0.0.1", port=7497, client_id=1))
    sig = Signal(symbol="AAPL", quantity=10, order_type="limit", limit_price=190.0)
    oid = broker.place_order(sig)
    open_orders = broker.get_open_orders()
    assert any(str(o.get("id")) == str(oid) for o in open_orders)
    assert broker.cancel_order(oid) is True
    open_orders2 = broker.get_open_orders()
    assert all(str(o.get("id")) != str(oid) or o.get("status") != "open" for o in open_orders2)


def test_ibkr_offline_account_positions_seed_and_get():
    broker = InteractiveBrokersBrokerAdapter()
    broker._seed_account(cash=100000.0, positions={"MSFT": {"quantity": 5, "avg_cost": 300.0}})
    acct = broker.get_account()
    assert acct["cash"] == 100000.0
    pos = broker.get_positions()
    assert pos["MSFT"]["quantity"] == 5
    assert pos["MSFT"]["avg_cost"] == 300.0


def test_ibkr_field_mapping_variants():
    brk = InteractiveBrokersBrokerAdapter()
    # Market
    t, f = brk._resolve_order_fields(Signal(symbol="AAPL", quantity=1))
    assert t == "MKT" and f == {}
    # Explicit Market
    t, f = brk._resolve_order_fields(Signal(symbol="AAPL", quantity=1, order_type="market"))
    assert t == "MKT"
    # Limit
    t, f = brk._resolve_order_fields(Signal(symbol="AAPL", quantity=1, order_type="limit", limit_price=123.45))
    assert t == "LMT" and "lmtPrice" in f
    # Stop
    t, f = brk._resolve_order_fields(Signal(symbol="AAPL", quantity=1, stop_price=120.0))
    assert t == "STP" and f.get("auxPrice") == 120.0
    # Stop Limit
    t, f = brk._resolve_order_fields(Signal(symbol="AAPL", quantity=1, stop_price=120.0, limit_price=121.0))
    assert t == "STP LMT" and set(f.keys()) == {"auxPrice", "lmtPrice"}


def test_ibkr_offline_subscribe_unsubscribe():
    brk = InteractiveBrokersBrokerAdapter()
    tid = brk.subscribe_market_data("AAPL")
    assert isinstance(tid, int)
    # Should create internal mapping allowing quote retrieval
    snap = brk._resp_book.get_quote_by_symbol("AAPL")
    assert snap == {} or isinstance(snap, dict)
    brk.unsubscribe_market_data("AAPL")
    assert brk._resp_book.get_quote_by_symbol("AAPL") is None


@pytest.mark.skipif(os.environ.get("HAS_IBAPI", "0") != "1", reason="ibapi not available in test env")
def test_ibkr_connect_disconnect_roundtrip():
    # This test is only enabled when HAS_IBAPI=1 and a gateway is reachable
    cfg = IBKRConfig.from_env()
    brk = InteractiveBrokersBrokerAdapter(cfg)
    try:
        brk.connect()
    finally:
        brk.disconnect()

