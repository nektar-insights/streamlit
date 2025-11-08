"""
Canonical Maturity Date Service
================================

This module provides the single source of truth for maturity date handling.
All maturity-related operations MUST go through this service to ensure:
- Traceability to HubSpot
- Consistent date handling
- Quality validation
- Audit trail maintenance

HubSpot is the sole source of truth for maturity dates.
"""

from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MaturityInfo:
    """
    Canonical maturity information for a loan.

    This dataclass represents the complete maturity metadata for a loan,
    including quality flags and audit trail.

    Attributes:
        loan_id: Unique identifier for the loan
        maturity_date: The actual maturity date (from HubSpot)
        maturity_basis: Origin of this maturity date
            - "original": Original deal maturity from HubSpot
            - "amended": Updated via amendment
            - "renewed": Deal was renewed
            - "extended": Maturity was extended
        maturity_version: Version number (increments with each change)
        maturity_last_updated_at: When this maturity date was last updated
        maturity_quality: Data quality flag
            - "ok": Valid maturity date
            - "missing": No maturity date provided
            - "invalid": Maturity date fails validation (e.g., before funding)
            - "backdated": Maturity date was backdated (regression)
        maturity_source: Original HubSpot field name
        amendment_id: Reference to amendment record (if applicable)
    """
    loan_id: str
    maturity_date: Optional[date]
    maturity_basis: str  # "original" | "amended" | "renewed" | "extended"
    maturity_version: int
    maturity_last_updated_at: datetime
    maturity_quality: str  # "ok" | "missing" | "invalid" | "backdated"
    maturity_source: str  # HubSpot field name
    amendment_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        d = asdict(self)
        # Convert date objects to ISO format strings
        if self.maturity_date:
            d['maturity_date'] = self.maturity_date.isoformat()
        if self.maturity_last_updated_at:
            d['maturity_last_updated_at'] = self.maturity_last_updated_at.isoformat()
        return d


class MaturityService:
    """
    Service for canonical maturity date operations.

    This service provides all maturity-related operations including:
    - Resolving maturity from HubSpot records
    - Validating maturity dates
    - Calculating days to/from maturity
    - Bucketing maturity dates
    - Selecting amendments deterministically
    """

    # HubSpot field mapping (configurable based on actual HubSpot schema)
    HUBSPOT_MATURITY_FIELD = "maturity_date"  # TODO: Verify actual HubSpot field name
    HUBSPOT_FUNDING_FIELD = "funding_date"

    @staticmethod
    def resolve_maturity(
        record: Dict[str, Any],
        amendments: Optional[List[Dict[str, Any]]] = None
    ) -> MaturityInfo:
        """
        Extract and resolve canonical maturity date from a deal record.

        This method implements the maturity resolution logic:
        1. Check for amendments (most recent takes precedence)
        2. Fall back to original deal maturity
        3. Validate the resulting date
        4. Determine quality flag

        Args:
            record: Deal record from HubSpot/Supabase
            amendments: Optional list of amendment records

        Returns:
            MaturityInfo: Complete maturity information with metadata
        """
        loan_id = str(record.get('loan_id', ''))
        funding_date = MaturityService._parse_date(record.get('funding_date'))

        # Step 1: Select amendment if available
        selected_amendment = None
        maturity_basis = "original"
        amendment_id = None

        if amendments and len(amendments) > 0:
            selected_amendment = MaturityService.select_amendment(amendments)
            if selected_amendment:
                maturity_date = MaturityService._parse_date(
                    selected_amendment.get('new_maturity_date') or
                    selected_amendment.get('maturity_date')
                )
                maturity_basis = selected_amendment.get('amendment_type', 'amended')
                amendment_id = selected_amendment.get('amendment_id')
                maturity_last_updated_at = MaturityService._parse_datetime(
                    selected_amendment.get('amendment_date') or
                    selected_amendment.get('updated_at')
                ) or datetime.now()

        # Step 2: Fall back to original maturity
        if not selected_amendment:
            maturity_date = MaturityService._parse_date(
                record.get('maturity_date') or
                record.get(MaturityService.HUBSPOT_MATURITY_FIELD)
            )
            maturity_last_updated_at = MaturityService._parse_datetime(
                record.get('updated_at') or
                record.get('last_modified_date')
            ) or datetime.now()

        # Step 3: Validate quality
        maturity_quality = MaturityService.validate_maturity(
            maturity_date,
            funding_date
        )

        # Step 4: Determine version (default to 1 if no amendments)
        maturity_version = len(amendments) if amendments else 1

        return MaturityInfo(
            loan_id=loan_id,
            maturity_date=maturity_date,
            maturity_basis=maturity_basis,
            maturity_version=maturity_version,
            maturity_last_updated_at=maturity_last_updated_at,
            maturity_quality=maturity_quality,
            maturity_source=MaturityService.HUBSPOT_MATURITY_FIELD,
            amendment_id=amendment_id
        )

    @staticmethod
    def select_amendment(amendments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Select the most recent valid amendment deterministically.

        Amendment precedence rules:
        1. Most recent amendment_date
        2. If dates are equal, highest amendment_id (lexicographic)
        3. Skip amendments with invalid/missing maturity dates

        Args:
            amendments: List of amendment records

        Returns:
            The selected amendment or None if no valid amendments
        """
        if not amendments:
            return None

        valid_amendments = []

        for amendment in amendments:
            # Check if amendment has a valid maturity date
            mat_date = MaturityService._parse_date(
                amendment.get('new_maturity_date') or
                amendment.get('maturity_date')
            )
            if mat_date:
                amendment_date = MaturityService._parse_datetime(
                    amendment.get('amendment_date') or
                    amendment.get('created_at')
                )
                if amendment_date:
                    valid_amendments.append(amendment)

        if not valid_amendments:
            return None

        # Sort by amendment_date (descending), then by amendment_id (descending)
        sorted_amendments = sorted(
            valid_amendments,
            key=lambda a: (
                MaturityService._parse_datetime(
                    a.get('amendment_date') or a.get('created_at')
                ),
                a.get('amendment_id', '')
            ),
            reverse=True
        )

        return sorted_amendments[0]

    @staticmethod
    def validate_maturity(
        maturity_date: Optional[date],
        funding_date: Optional[date] = None
    ) -> str:
        """
        Validate maturity date and return quality flag.

        Validation rules:
        - "missing": maturity_date is None
        - "invalid": maturity_date <= funding_date (if funding_date provided)
        - "ok": maturity_date is valid

        Args:
            maturity_date: The maturity date to validate
            funding_date: The funding date (for validation)

        Returns:
            Quality flag: "ok" | "missing" | "invalid"
        """
        if maturity_date is None:
            return "missing"

        if funding_date is not None:
            if maturity_date <= funding_date:
                logger.warning(
                    f"Invalid maturity: maturity_date ({maturity_date}) <= "
                    f"funding_date ({funding_date})"
                )
                return "invalid"

        return "ok"

    @staticmethod
    def days_to_maturity(
        as_of: date,
        maturity_date: Optional[date]
    ) -> Optional[int]:
        """
        Calculate days from as_of date to maturity.

        Returns positive number for future maturity,
        negative number for past maturity.

        Args:
            as_of: Reference date (usually today)
            maturity_date: The maturity date

        Returns:
            Number of days, or None if maturity_date is None
        """
        if maturity_date is None:
            return None

        delta = maturity_date - as_of
        return delta.days

    @staticmethod
    def bucket_maturity(days: Optional[int]) -> str:
        """
        Bucket maturity into standard time ranges.

        Buckets:
        - "past": days < 0 (already matured)
        - "0-30": 0 <= days <= 30
        - "31-60": 31 <= days <= 60
        - "61-90": 61 <= days <= 90
        - ">90": days > 90
        - "unknown": days is None

        Args:
            days: Number of days to maturity

        Returns:
            Bucket label
        """
        if days is None:
            return "unknown"

        if days < 0:
            return "past"
        elif days <= 30:
            return "0-30"
        elif days <= 60:
            return "31-60"
        elif days <= 90:
            return "61-90"
        else:
            return ">90"

    @staticmethod
    def calculate_days_past_maturity(
        as_of: date,
        maturity_date: Optional[date]
    ) -> int:
        """
        Calculate days past maturity (0 if not yet matured).

        Used for risk scoring and delinquency calculations.

        Args:
            as_of: Reference date (usually today)
            maturity_date: The maturity date

        Returns:
            Number of days past maturity (0 if not yet matured or unknown)
        """
        if maturity_date is None:
            return 0

        if as_of <= maturity_date:
            return 0  # Not yet matured

        delta = as_of - maturity_date
        return max(0, delta.days)

    @staticmethod
    def calculate_remaining_maturity_months(
        as_of: date,
        maturity_date: Optional[date],
        only_active: bool = True
    ) -> float:
        """
        Calculate remaining months to maturity.

        Args:
            as_of: Reference date (usually today)
            maturity_date: The maturity date
            only_active: If True, return 0 for past maturity dates

        Returns:
            Number of months remaining (0 if matured or unknown)
        """
        if maturity_date is None:
            return 0.0

        if only_active and as_of >= maturity_date:
            return 0.0  # Already matured

        delta = maturity_date - as_of
        months = delta.days / 30.0

        return max(0.0, months)

    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        """Parse various date formats to date object"""
        if value is None:
            return None

        if isinstance(value, date):
            return value

        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, str):
            try:
                return pd.to_datetime(value).date()
            except:
                logger.warning(f"Failed to parse date: {value}")
                return None

        if isinstance(value, pd.Timestamp):
            return value.date()

        return None

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        """Parse various datetime formats to datetime object"""
        if value is None:
            return None

        if isinstance(value, datetime):
            return value

        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())

        if isinstance(value, str):
            try:
                return pd.to_datetime(value).to_pydatetime()
            except:
                logger.warning(f"Failed to parse datetime: {value}")
                return None

        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()

        return None


# Module-level convenience functions for easy import
def get_maturity_info(
    record: Dict[str, Any],
    amendments: Optional[List[Dict[str, Any]]] = None
) -> MaturityInfo:
    """Convenience function to get maturity info"""
    return MaturityService.resolve_maturity(record, amendments)


def days_to_maturity(as_of: date, maturity_date: Optional[date]) -> Optional[int]:
    """Convenience function to calculate days to maturity"""
    return MaturityService.days_to_maturity(as_of, maturity_date)


def bucket_maturity(days: Optional[int]) -> str:
    """Convenience function to bucket maturity"""
    return MaturityService.bucket_maturity(days)


def validate_maturity(
    maturity_date: Optional[date],
    funding_date: Optional[date] = None
) -> str:
    """Convenience function to validate maturity"""
    return MaturityService.validate_maturity(maturity_date, funding_date)
