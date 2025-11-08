"""
Unit tests for canonical maturity service.

Tests cover:
- Maturity date resolution
- Amendment selection
- Validation logic
- Days to maturity calculations
- Maturity bucketing
- Edge cases and error handling
"""

import pytest
from datetime import date, datetime, timedelta
from services.maturity import MaturityService, MaturityInfo


class TestMaturityServiceValidation:
    """Test maturity date validation logic"""

    def test_validate_maturity_ok(self):
        """Test valid maturity date"""
        maturity_date = date(2025, 12, 31)
        funding_date = date(2024, 1, 1)
        result = MaturityService.validate_maturity(maturity_date, funding_date)
        assert result == "ok"

    def test_validate_maturity_missing(self):
        """Test missing maturity date"""
        result = MaturityService.validate_maturity(None, date(2024, 1, 1))
        assert result == "missing"

    def test_validate_maturity_invalid_before_funding(self):
        """Test maturity date before funding date"""
        maturity_date = date(2024, 1, 1)
        funding_date = date(2024, 6, 1)
        result = MaturityService.validate_maturity(maturity_date, funding_date)
        assert result == "invalid"

    def test_validate_maturity_invalid_equal_funding(self):
        """Test maturity date equal to funding date"""
        same_date = date(2024, 6, 1)
        result = MaturityService.validate_maturity(same_date, same_date)
        assert result == "invalid"

    def test_validate_maturity_no_funding_date(self):
        """Test validation without funding date"""
        maturity_date = date(2025, 12, 31)
        result = MaturityService.validate_maturity(maturity_date, None)
        assert result == "ok"


class TestMaturityServiceDaysCalculations:
    """Test days to/from maturity calculations"""

    def test_days_to_maturity_future(self):
        """Test days to future maturity"""
        today = date(2024, 1, 1)
        maturity = date(2024, 2, 1)
        result = MaturityService.days_to_maturity(today, maturity)
        assert result == 31

    def test_days_to_maturity_past(self):
        """Test days to past maturity (negative)"""
        today = date(2024, 2, 1)
        maturity = date(2024, 1, 1)
        result = MaturityService.days_to_maturity(today, maturity)
        assert result == -31

    def test_days_to_maturity_none(self):
        """Test days to maturity with None date"""
        today = date(2024, 1, 1)
        result = MaturityService.days_to_maturity(today, None)
        assert result is None

    def test_days_past_maturity_overdue(self):
        """Test days past maturity for overdue loan"""
        today = date(2024, 2, 1)
        maturity = date(2024, 1, 1)
        result = MaturityService.calculate_days_past_maturity(today, maturity)
        assert result == 31

    def test_days_past_maturity_not_yet_matured(self):
        """Test days past maturity for active loan"""
        today = date(2024, 1, 1)
        maturity = date(2024, 2, 1)
        result = MaturityService.calculate_days_past_maturity(today, maturity)
        assert result == 0

    def test_days_past_maturity_none(self):
        """Test days past maturity with None date"""
        today = date(2024, 1, 1)
        result = MaturityService.calculate_days_past_maturity(today, None)
        assert result == 0


class TestMaturityServiceBucketing:
    """Test maturity date bucketing"""

    def test_bucket_maturity_past(self):
        """Test bucketing for past maturity"""
        assert MaturityService.bucket_maturity(-10) == "past"
        assert MaturityService.bucket_maturity(-1) == "past"

    def test_bucket_maturity_0_to_30(self):
        """Test bucketing for 0-30 days"""
        assert MaturityService.bucket_maturity(0) == "0-30"
        assert MaturityService.bucket_maturity(15) == "0-30"
        assert MaturityService.bucket_maturity(30) == "0-30"

    def test_bucket_maturity_31_to_60(self):
        """Test bucketing for 31-60 days"""
        assert MaturityService.bucket_maturity(31) == "31-60"
        assert MaturityService.bucket_maturity(45) == "31-60"
        assert MaturityService.bucket_maturity(60) == "31-60"

    def test_bucket_maturity_61_to_90(self):
        """Test bucketing for 61-90 days"""
        assert MaturityService.bucket_maturity(61) == "61-90"
        assert MaturityService.bucket_maturity(75) == "61-90"
        assert MaturityService.bucket_maturity(90) == "61-90"

    def test_bucket_maturity_over_90(self):
        """Test bucketing for >90 days"""
        assert MaturityService.bucket_maturity(91) == ">90"
        assert MaturityService.bucket_maturity(365) == ">90"

    def test_bucket_maturity_none(self):
        """Test bucketing for None"""
        assert MaturityService.bucket_maturity(None) == "unknown"


class TestMaturityServiceRemainingMonths:
    """Test remaining maturity months calculation"""

    def test_remaining_maturity_months_active(self):
        """Test remaining months for active loan"""
        today = date(2024, 1, 1)
        maturity = date(2024, 7, 1)  # 6 months away (180 days)
        result = MaturityService.calculate_remaining_maturity_months(today, maturity)
        assert result == pytest.approx(6.0, abs=0.1)

    def test_remaining_maturity_months_past(self):
        """Test remaining months for past maturity (should return 0)"""
        today = date(2024, 7, 1)
        maturity = date(2024, 1, 1)
        result = MaturityService.calculate_remaining_maturity_months(today, maturity, only_active=True)
        assert result == 0.0

    def test_remaining_maturity_months_none(self):
        """Test remaining months with None maturity"""
        today = date(2024, 1, 1)
        result = MaturityService.calculate_remaining_maturity_months(today, None)
        assert result == 0.0


class TestMaturityServiceAmendmentSelection:
    """Test amendment selection logic"""

    def test_select_amendment_most_recent(self):
        """Test selecting most recent amendment"""
        amendments = [
            {
                'amendment_id': 'A1',
                'amendment_date': datetime(2024, 1, 1),
                'new_maturity_date': date(2025, 1, 1)
            },
            {
                'amendment_id': 'A2',
                'amendment_date': datetime(2024, 6, 1),
                'new_maturity_date': date(2025, 6, 1)
            },
            {
                'amendment_id': 'A3',
                'amendment_date': datetime(2024, 3, 1),
                'new_maturity_date': date(2025, 3, 1)
            }
        ]
        result = MaturityService.select_amendment(amendments)
        assert result['amendment_id'] == 'A2'

    def test_select_amendment_skip_invalid(self):
        """Test skipping amendments with invalid maturity dates"""
        amendments = [
            {
                'amendment_id': 'A1',
                'amendment_date': datetime(2024, 1, 1),
                'new_maturity_date': date(2025, 1, 1)
            },
            {
                'amendment_id': 'A2',
                'amendment_date': datetime(2024, 6, 1),
                'new_maturity_date': None  # Invalid
            }
        ]
        result = MaturityService.select_amendment(amendments)
        assert result['amendment_id'] == 'A1'

    def test_select_amendment_empty_list(self):
        """Test selecting from empty amendment list"""
        result = MaturityService.select_amendment([])
        assert result is None

    def test_select_amendment_none(self):
        """Test selecting from None"""
        result = MaturityService.select_amendment(None)
        assert result is None

    def test_select_amendment_determinism(self):
        """Test that amendment selection is deterministic"""
        amendments = [
            {
                'amendment_id': 'A1',
                'amendment_date': datetime(2024, 1, 1),
                'new_maturity_date': date(2025, 1, 1)
            },
            {
                'amendment_id': 'A2',
                'amendment_date': datetime(2024, 1, 1),  # Same date as A1
                'new_maturity_date': date(2025, 2, 1)
            }
        ]
        # Should select A2 (lexicographically higher ID)
        result = MaturityService.select_amendment(amendments)
        assert result['amendment_id'] == 'A2'


class TestMaturityServiceResolveMaturity:
    """Test maturity resolution from records"""

    def test_resolve_maturity_original(self):
        """Test resolving original maturity (no amendments)"""
        record = {
            'loan_id': 'L001',
            'maturity_date': date(2025, 12, 31),
            'funding_date': date(2024, 1, 1),
            'updated_at': datetime(2024, 1, 1, 12, 0)
        }
        result = MaturityService.resolve_maturity(record)

        assert result.loan_id == 'L001'
        assert result.maturity_date == date(2025, 12, 31)
        assert result.maturity_basis == 'original'
        assert result.maturity_quality == 'ok'
        assert result.amendment_id is None

    def test_resolve_maturity_with_amendment(self):
        """Test resolving maturity with amendment"""
        record = {
            'loan_id': 'L002',
            'maturity_date': date(2025, 6, 1),
            'funding_date': date(2024, 1, 1),
            'updated_at': datetime(2024, 1, 1, 12, 0)
        }
        amendments = [
            {
                'amendment_id': 'A001',
                'amendment_date': datetime(2024, 6, 1),
                'amendment_type': 'extended',
                'new_maturity_date': date(2025, 12, 31),
                'updated_at': datetime(2024, 6, 1, 12, 0)
            }
        ]
        result = MaturityService.resolve_maturity(record, amendments)

        assert result.loan_id == 'L002'
        assert result.maturity_date == date(2025, 12, 31)
        assert result.maturity_basis == 'extended'
        assert result.amendment_id == 'A001'

    def test_resolve_maturity_invalid(self):
        """Test resolving invalid maturity (before funding)"""
        record = {
            'loan_id': 'L003',
            'maturity_date': date(2024, 1, 1),
            'funding_date': date(2024, 6, 1),
            'updated_at': datetime(2024, 1, 1, 12, 0)
        }
        result = MaturityService.resolve_maturity(record)

        assert result.loan_id == 'L003'
        assert result.maturity_quality == 'invalid'

    def test_resolve_maturity_missing(self):
        """Test resolving missing maturity"""
        record = {
            'loan_id': 'L004',
            'maturity_date': None,
            'funding_date': date(2024, 1, 1),
            'updated_at': datetime(2024, 1, 1, 12, 0)
        }
        result = MaturityService.resolve_maturity(record)

        assert result.loan_id == 'L004'
        assert result.maturity_date is None
        assert result.maturity_quality == 'missing'


class TestMaturityServiceDateParsing:
    """Test date parsing utilities"""

    def test_parse_date_from_date(self):
        """Test parsing date object"""
        d = date(2024, 1, 1)
        result = MaturityService._parse_date(d)
        assert result == d

    def test_parse_date_from_datetime(self):
        """Test parsing datetime object"""
        dt = datetime(2024, 1, 1, 12, 30)
        result = MaturityService._parse_date(dt)
        assert result == date(2024, 1, 1)

    def test_parse_date_from_string(self):
        """Test parsing date string"""
        s = "2024-01-01"
        result = MaturityService._parse_date(s)
        assert result == date(2024, 1, 1)

    def test_parse_date_from_none(self):
        """Test parsing None"""
        result = MaturityService._parse_date(None)
        assert result is None

    def test_parse_date_from_invalid_string(self):
        """Test parsing invalid string"""
        result = MaturityService._parse_date("invalid")
        assert result is None


class TestMaturityServiceEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_leap_day_maturity(self):
        """Test leap day handling"""
        today = date(2024, 2, 28)
        maturity = date(2024, 2, 29)  # Leap day
        result = MaturityService.days_to_maturity(today, maturity)
        assert result == 1

    def test_extreme_future_date(self):
        """Test extremely far future maturity"""
        today = date(2024, 1, 1)
        maturity = date(2099, 12, 31)
        result = MaturityService.days_to_maturity(today, maturity)
        assert result > 0

    def test_extreme_past_date(self):
        """Test extremely far past maturity"""
        today = date(2024, 1, 1)
        maturity = date(1900, 1, 1)
        result = MaturityService.days_to_maturity(today, maturity)
        assert result < 0

    def test_same_day_maturity(self):
        """Test maturity on the same day"""
        today = date(2024, 1, 1)
        result = MaturityService.days_to_maturity(today, today)
        assert result == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
