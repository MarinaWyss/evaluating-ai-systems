"""A tiny fake BeefCake store: orders, returns, warranties, and support tickets.

The v2 and v3 bots change this data through tools. Tests reset it, run the
agent, and then check the END STATE (was a return created?) rather than the
exact sequence of tool calls. Everything is in memory and deterministic.
"""

from __future__ import annotations

import copy
from datetime import date

# The store's "today", fixed so return windows and warranties behave the same every run.
TODAY = date(2026, 10, 5)

RETURN_WINDOW_DAYS = 30
WARRANTY_YEARS = {"BeefCake Row": 2, "BeefCake Bell": 2, "BeefCake Pulse": 1}

ORDERS = {
    "BC-4417": {"customer_email": "sam.k@example.com", "items": ["BeefCake Bell"],
                "status": "delivered", "delivered_on": "2026-09-20", "arrived_damaged": False},
    "BC-4420": {"customer_email": "priya.n@example.com", "items": ["BeefCake Row"],
                "status": "delivered", "delivered_on": "2026-08-14", "arrived_damaged": False},
    "BC-4431": {"customer_email": "leo.m@example.com", "items": ["BeefCake Pulse", "BeefCake Bell"],
                "status": "delivered", "delivered_on": "2026-09-28", "arrived_damaged": False},
    "BC-4452": {"customer_email": "ana.r@example.com", "items": ["BeefCake Bell"],
                "status": "delivered", "delivered_on": "2026-10-02", "arrived_damaged": True},
    "BC-4460": {"customer_email": "jo.t@example.com", "items": ["BeefCake Row", "BeefCake Pulse"],
                "status": "in transit", "delivered_on": None, "arrived_damaged": False},
}

DEVICES = {
    "ROW-2025-00318": {"product": "BeefCake Row", "delivered_on": "2025-03-02"},
    "ROW-2024-00871": {"product": "BeefCake Row", "delivered_on": "2024-06-11"},
    "BELL-2026-01944": {"product": "BeefCake Bell", "delivered_on": "2026-09-20"},
    "PULSE-2025-00562": {"product": "BeefCake Pulse", "delivered_on": "2025-05-30"},
}


class Store:
    """The fake store's current state. Call reset() before each test run."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.orders = copy.deepcopy(ORDERS)
        self.returns: list[dict] = []
        self.tickets: list[dict] = []

    def snapshot(self) -> dict:
        """The end state that tests compare against."""
        return {
            "returns": [{"order_id": r["order_id"], "item": r["item"]} for r in self.returns],
            "tickets": len(self.tickets),
        }


STORE = Store()


def _days_since(iso: str) -> int:
    return (TODAY - date.fromisoformat(iso)).days


def lookup_order(order_id: str, store: Store = STORE) -> dict:
    order = store.orders.get(order_id)
    if order is None:
        return {"error": f"No order found with ID {order_id!r}. Order IDs look like BC-1234."}
    return {"order_id": order_id, **{k: v for k, v in order.items() if k != "arrived_damaged"}}


def start_return(order_id: str, item: str, reason: str, store: Store = STORE) -> dict:
    order = store.orders.get(order_id)
    if order is None:
        return {"error": f"No order found with ID {order_id!r}."}
    if item not in order["items"]:
        return {"error": f"{item!r} isn't in order {order_id}. Items: {', '.join(order['items'])}."}
    if order["delivered_on"] is None:
        return {"error": f"Order {order_id} hasn't been delivered yet."}
    days = _days_since(order["delivered_on"])
    if days > RETURN_WINDOW_DAYS:
        return {"error": f"Order {order_id} was delivered {days} days ago, outside the {RETURN_WINDOW_DAYS}-day return window."}
    record = {
        "return_id": f"R-{len(store.returns) + 1:03d}",
        "order_id": order_id,
        "item": item,
        "reason": reason,
        "prepaid_label": order["arrived_damaged"],
    }
    store.returns.append(record)
    return record


def check_warranty(serial_number: str, store: Store = STORE) -> dict:
    device = DEVICES.get(serial_number)
    if device is None:
        return {"error": f"No device found with serial number {serial_number!r}."}
    years = WARRANTY_YEARS[device["product"]]
    start = date.fromisoformat(device["delivered_on"])
    covered_until = start.replace(year=start.year + years)
    return {
        "serial_number": serial_number,
        "product": device["product"],
        "delivered_on": device["delivered_on"],
        "warranty_years": years,
        "covered_until": covered_until.isoformat(),
        "in_warranty": TODAY <= covered_until,
    }


def create_ticket(summary: str, store: Store = STORE) -> dict:
    ticket = {"ticket_id": f"T-{len(store.tickets) + 1:03d}", "summary": summary}
    store.tickets.append(ticket)
    return {**ticket, "message": "A BeefCake Support agent will reply by email within one business day."}
