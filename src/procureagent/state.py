"""Immutable supplier lookup and AP lifecycle transitions.

The fixture is deliberately a tiny in-memory database.  It is keyed by the
full ``(supplier_id, invoice_number)`` identity; an invoice number by itself is
never enough to activate an obligation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .contracts import (
    ContractValidationError,
    InvoiceIdentity,
    InvoicePaymentStatus,
    PaymentProof,
    RestaurantState,
    SyntheticSupplierInvoice,
    VerifiedInvoiceIdentity,
    validate_full_payment_proof,
)


class StateTransitionError(ContractValidationError):
    """Raised when a requested ledger transition is unsafe or stale."""


def invoice_index(
    state: RestaurantState,
) -> dict[InvoiceIdentity, SyntheticSupplierInvoice]:
    """Return a fresh exact-composite-key index for an immutable state."""

    if not isinstance(state, RestaurantState):
        raise StateTransitionError("state must be RestaurantState")
    return {invoice.identity: invoice for invoice in state.invoices}


def lookup_invoice(
    state: RestaurantState, identity: InvoiceIdentity
) -> SyntheticSupplierInvoice | None:
    """Look up one exact supplier-and-invoice identity without side effects."""

    if not isinstance(identity, InvoiceIdentity):
        raise StateTransitionError("lookup requires a complete InvoiceIdentity")
    return invoice_index(state).get(identity)


def lookup_verified_invoice(
    state: RestaurantState, verified_identity: VerifiedInvoiceIdentity
) -> SyntheticSupplierInvoice | None:
    """Activate lookup only for an identity that passed the document gate."""

    if not isinstance(verified_identity, VerifiedInvoiceIdentity):
        raise StateTransitionError("lookup requires VerifiedInvoiceIdentity")
    return lookup_invoice(state, verified_identity.identity)


def require_invoice(
    state: RestaurantState, identity: InvoiceIdentity
) -> SyntheticSupplierInvoice:
    """Return an exact match or fail closed."""

    invoice = lookup_invoice(state, identity)
    if invoice is None:
        raise StateTransitionError("unknown composite supplier/invoice identity")
    return invoice


def rebuild_versioned_state(
    state: RestaurantState,
    *,
    invoices: Iterable[SyntheticSupplierInvoice],
    cash_minor: int | None = None,
    day: int | None = None,
) -> RestaurantState:
    """Create the next immutable state and synchronize every record version.

    This helper is for validated domain transitions.  Contracts revalidate all
    values, including integer cents and the shared state-version invariant.
    """

    if not isinstance(state, RestaurantState):
        raise StateTransitionError("state must be RestaurantState")
    next_version = state.state_version + 1
    versioned = tuple(
        replace(invoice, state_version=next_version) for invoice in tuple(invoices)
    )
    return replace(
        state,
        day=state.day if day is None else day,
        state_version=next_version,
        cash_minor=state.cash_minor if cash_minor is None else cash_minor,
        invoices=versioned,
    )


def close_invoice_with_payment_proof(
    state: RestaurantState,
    proof: PaymentProof,
    *,
    consumed_receipt_ids: Iterable[str] = (),
) -> RestaurantState:
    """Confirm a simulated AP payment using one unused, exact full proof.

    The contract validator enforces the only legal lifecycle edge here:
    ``simulated_payment_approved -> paid_confirmed``.  A failed check returns no
    partial state and does not consume the receipt ID.
    """

    if not isinstance(proof, PaymentProof):
        raise StateTransitionError("proof must be PaymentProof")
    consumed = tuple(consumed_receipt_ids)
    if any(not isinstance(value, str) for value in consumed):
        raise StateTransitionError("consumed receipt IDs must be text")
    if len(consumed) != len(set(consumed)):
        raise StateTransitionError("consumed receipt IDs cannot contain duplicates")
    if proof.receipt_id in consumed:
        raise StateTransitionError("payment proof receipt ID was already consumed")

    invoice = require_invoice(state, proof.identity)
    validate_full_payment_proof(invoice, proof)

    updated = tuple(
        replace(item, payment_status=InvoicePaymentStatus.PAID_CONFIRMED)
        if item.identity == proof.identity
        else item
        for item in state.invoices
    )
    return rebuild_versioned_state(state, invoices=updated)


__all__ = [
    "StateTransitionError",
    "close_invoice_with_payment_proof",
    "invoice_index",
    "lookup_invoice",
    "lookup_verified_invoice",
    "rebuild_versioned_state",
    "require_invoice",
]
