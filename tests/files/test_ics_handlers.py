import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import TestCase

import pandas as pd
from icalendar import Calendar, vRecur

from mitoolspro.files.ics_handlers import (
    convert_to_dataframe,
    count_events_by_date,
    extract_events,
    format_event_for_display,
    get_events_between_dates,
    get_unique_attendees,
    get_unique_organizers,
    read_ics_file,
)


class TestICSHandler(TestCase):
    def setUp(self):
        self.sample_ics_content = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
SUMMARY:Test Event 1
DESCRIPTION:Test Description 1
DTSTART:20240101T100000Z
DTEND:20240101T110000Z
ORGANIZER:mailto:organizer1@example.com
ATTENDEE:mailto:attendee1@example.com
ATTENDEE:mailto:attendee2@example.com
URL:https://example.com
UID:123456
TRANSP:OPAQUE
STATUS:CONFIRMED
SEQUENCE:0
LOCATION:Test Location
DTSTAMP:20240101T090000Z
CREATED:20240101T080000Z
CLASS:PUBLIC
END:VEVENT
BEGIN:VEVENT
SUMMARY:Test Event 2
DTSTART:20240102T100000Z
DTEND:20240102T110000Z
ORGANIZER:mailto:organizer2@example.com
ATTENDEE:mailto:attendee3@example.com
END:VEVENT
END:VCALENDAR"""

        self.complex_ics_content = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
SUMMARY:All Day Event
DTSTART;VALUE=DATE:20240201
DTEND;VALUE=DATE:20240202
ORGANIZER:mailto:organizer3@example.com
STATUS:TENTATIVE
CLASS:PRIVATE
END:VEVENT
BEGIN:VEVENT
SUMMARY:Recurring Meeting
DTSTART:20240105T140000Z
DTEND:20240105T150000Z
RRULE:FREQ=WEEKLY;COUNT=4
ORGANIZER:mailto:organizer4@example.com
LOCATION:Virtual Room 1
END:VEVENT
BEGIN:VEVENT
SUMMARY:Multi-day Conference
DTSTART:20240115T090000Z
DTEND:20240117T170000Z
ORGANIZER:mailto:organizer5@example.com
LOCATION:Convention Center
DESCRIPTION:Three day conference with multiple sessions
ATTENDEE:mailto:speaker1@example.com
ATTENDEE:mailto:speaker2@example.com
ATTENDEE:mailto:attendee4@example.com
STATUS:CONFIRMED
URL:https://conference.example.com
END:VEVENT
BEGIN:VEVENT
SUMMARY:Cancelled Meeting
DTSTART:20240120T100000Z
DTEND:20240120T110000Z
STATUS:CANCELLED
ORGANIZER:mailto:organizer1@example.com
END:VEVENT
BEGIN:VEVENT
SUMMARY:Meeting with Special Characters
DESCRIPTION:Test with émojis 🎉 and üñîçødé
DTSTART:20240125T100000Z
DTEND:20240125T110000Z
LOCATION:Café & Restaurant
ORGANIZER:mailto:organizer6@example.com
END:VEVENT
END:VCALENDAR"""

        self.temp_dir = tempfile.TemporaryDirectory()
        self.ics_path = Path(self.temp_dir.name) / "test.ics"
        self.complex_ics_path = Path(self.temp_dir.name) / "complex_test.ics"
        self.ics_path.write_text(self.sample_ics_content)
        self.complex_ics_path.write_text(self.complex_ics_content)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_ics_file(self):
        calendar = read_ics_file(self.ics_path)
        self.assertIsInstance(calendar, Calendar)
        self.assertEqual(len(list(calendar.walk("VEVENT"))), 2)

        complex_calendar = read_ics_file(self.complex_ics_path)
        self.assertEqual(len(list(complex_calendar.walk("VEVENT"))), 5)

        with self.assertRaises(FileNotFoundError):
            read_ics_file(Path(self.temp_dir.name) / "nonexistent.ics")

    def test_extract_events_basic(self):
        calendar = read_ics_file(self.ics_path)
        events = extract_events(calendar)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["summary"], "Test Event 1")
        self.assertEqual(events[0]["description"], "Test Description 1")
        self.assertEqual(events[0]["organizer"], "organizer1@example.com")
        self.assertEqual(
            events[0]["attendees"],
            ["attendee1@example.com", "attendee2@example.com"],
        )
        self.assertEqual(events[0]["location"], "Test Location")
        self.assertEqual(events[0]["url"], "https://example.com")

        self.assertEqual(events[1]["summary"], "Test Event 2")
        self.assertEqual(events[1]["organizer"], "organizer2@example.com")
        self.assertEqual(events[1]["attendees"], ["attendee3@example.com"])

    def test_extract_events_complex(self):
        calendar = read_ics_file(self.complex_ics_path)
        events = extract_events(calendar)

        self.assertEqual(len(events), 5)

        all_day_event = next(e for e in events if e["summary"] == "All Day Event")
        self.assertEqual(all_day_event["status"], "TENTATIVE")
        self.assertEqual(all_day_event["class"], "PRIVATE")
        self.assertIsInstance(all_day_event["start"], pd.Timestamp)
        self.assertIsInstance(all_day_event["end"], pd.Timestamp)

        recurring_event = next(e for e in events if e["summary"] == "Recurring Meeting")
        self.assertEqual(recurring_event["location"], "Virtual Room 1")
        self.assertIsInstance(recurring_event["rrule"], vRecur)
        self.assertEqual(recurring_event["rrule"]["FREQ"], ["WEEKLY"])
        self.assertEqual(recurring_event["rrule"]["COUNT"], [4])

        conference = next(e for e in events if e["summary"] == "Multi-day Conference")
        self.assertEqual(len(conference["attendees"]), 3)
        self.assertEqual(conference["url"], "https://conference.example.com")
        duration = conference["end"] - conference["start"]
        self.assertEqual(duration.days, 2)

        cancelled_meeting = next(
            e for e in events if e["summary"] == "Cancelled Meeting"
        )
        self.assertEqual(cancelled_meeting["status"], "CANCELLED")

        special_chars = next(
            e for e in events if e["summary"] == "Meeting with Special Characters"
        )
        self.assertIn("émojis 🎉", special_chars["description"])
        self.assertEqual(special_chars["location"], "Café & Restaurant")

    def test_count_events_by_date(self):
        calendar = read_ics_file(self.complex_ics_path)
        events = extract_events(calendar)
        event_counts = count_events_by_date(events)

        print(event_counts)

        self.assertEqual(len(event_counts), 5)
        self.assertEqual(event_counts["2024-02-01"], 1)  # All Day Event
        self.assertEqual(event_counts["2024-01-05"], 1)  # Recurring Meeting
        self.assertEqual(event_counts["2024-01-15"], 1)  # Multi-day Conference Day 1
        self.assertEqual(event_counts["2024-01-20"], 1)  # Cancelled Meeting
        self.assertEqual(
            event_counts["2024-01-25"], 1
        )  # Meeting with Special Characters

    def test_get_unique_organizers(self):
        calendar = read_ics_file(self.complex_ics_path)
        events = extract_events(calendar)
        organizers = get_unique_organizers(events)

        self.assertEqual(len(organizers), 5)
        expected_organizers = {
            "organizer1@example.com",
            "organizer3@example.com",
            "organizer4@example.com",
            "organizer5@example.com",
            "organizer6@example.com",
        }
        self.assertTrue(expected_organizers.issubset(organizers))

    def test_get_unique_attendees(self):
        calendar = read_ics_file(self.complex_ics_path)
        events = extract_events(calendar)
        attendees = get_unique_attendees(events)

        self.assertEqual(len(attendees), 3)
        expected_attendees = {
            "speaker1@example.com",
            "speaker2@example.com",
            "attendee4@example.com",
        }
        self.assertTrue(expected_attendees.issubset(attendees))

    def test_convert_to_dataframe(self):
        calendar = read_ics_file(self.complex_ics_path)
        events = extract_events(calendar)
        df = convert_to_dataframe(events)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        self.assertTrue(
            all(
                col in df.columns
                for col in [
                    "summary",
                    "description",
                    "start",
                    "end",
                    "organizer",
                    "attendees",
                    "location",
                    "status",
                    "url",
                ]
            )
        )
        self.assertTrue(
            all(
                isinstance(dt, pd.Timestamp) or dt is None
                for dt in df["start"]
                if dt is not None
            )
        )
        self.assertTrue(
            all(
                isinstance(dt, pd.Timestamp) or dt is None
                for dt in df["end"]
                if dt is not None
            )
        )
        self.assertTrue(df["attendees"].apply(lambda x: isinstance(x, list)).all())

    def test_get_events_between_dates(self):
        calendar = read_ics_file(self.complex_ics_path)
        events = extract_events(calendar)

        start_date = datetime(2024, 1, 15)
        end_date = datetime(2024, 1, 17)
        filtered_events = get_events_between_dates(events, start_date, end_date)
        self.assertEqual(len(filtered_events), 1)
        self.assertEqual(filtered_events[0]["summary"], "Multi-day Conference")
        self.assertEqual(filtered_events[0]["location"], "Convention Center")

        # Test multi-day event - partial overlap at start
        start_date = datetime(2024, 1, 14)
        end_date = datetime(2024, 1, 15)
        filtered_events = get_events_between_dates(events, start_date, end_date)
        self.assertEqual(len(filtered_events), 1)
        self.assertEqual(filtered_events[0]["summary"], "Multi-day Conference")

        # Test multi-day event - partial overlap at end
        start_date = datetime(2024, 1, 17)
        end_date = datetime(2024, 1, 18)
        filtered_events = get_events_between_dates(events, start_date, end_date)
        self.assertEqual(len(filtered_events), 1)
        self.assertEqual(filtered_events[0]["summary"], "Multi-day Conference")

        # Test multi-day event - middle day
        start_date = datetime(2024, 1, 16)
        end_date = datetime(2024, 1, 16)
        filtered_events = get_events_between_dates(events, start_date, end_date)
        self.assertEqual(len(filtered_events), 1)
        self.assertEqual(filtered_events[0]["summary"], "Multi-day Conference")

        # Test all-day event
        start_date = datetime(2024, 2, 1)
        end_date = datetime(2024, 2, 1)
        filtered_events = get_events_between_dates(events, start_date, end_date)
        self.assertEqual(len(filtered_events), 1)
        self.assertEqual(filtered_events[0]["summary"], "All Day Event")

        # Test date range with no events
        start_date = datetime(2024, 3, 1)
        end_date = datetime(2024, 3, 31)
        filtered_events = get_events_between_dates(events, start_date, end_date)
        self.assertEqual(len(filtered_events), 0)

        # Test multiple events in range
        start_date = datetime(2024, 1, 15)
        end_date = datetime(2024, 1, 25)
        filtered_events = get_events_between_dates(events, start_date, end_date)
        self.assertEqual(
            len(filtered_events), 3
        )  # Conference, Cancelled Meeting, and Special Characters
        summaries = {event["summary"] for event in filtered_events}
        self.assertEqual(
            summaries,
            {
                "Multi-day Conference",
                "Cancelled Meeting",
                "Meeting with Special Characters",
            },
        )

    def test_format_event_for_display(self):
        calendar = read_ics_file(self.complex_ics_path)
        events = extract_events(calendar)

        conference_event = next(
            e for e in events if e["summary"] == "Multi-day Conference"
        )
        formatted_event = format_event_for_display(conference_event)

        self.assertIn("Summary: Multi-day Conference", formatted_event)
        self.assertIn(
            "Description: Three day conference with multiple sessions", formatted_event
        )
        self.assertIn(
            "speaker1@example.com, speaker2@example.com, attendee4@example.com",
            formatted_event,
        )
        self.assertIn("Organizer: organizer5@example.com", formatted_event)

        special_chars_event = next(
            e for e in events if "Special Characters" in e["summary"]
        )
        formatted_special = format_event_for_display(special_chars_event)
        self.assertIn("émojis 🎉", formatted_special)
        self.assertIn("Café & Restaurant", formatted_special)

    def test_edge_cases(self):
        empty_calendar = Calendar()
        events = extract_events(empty_calendar)
        self.assertEqual(len(events), 0)

        event_counts = count_events_by_date(events)
        self.assertEqual(len(event_counts), 0)

        organizers = get_unique_organizers(events)
        self.assertEqual(len(organizers), 0)

        attendees = get_unique_attendees(events)
        self.assertEqual(len(attendees), 0)

        df = convert_to_dataframe(events)
        self.assertEqual(len(df), 0)

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        filtered_events = get_events_between_dates(events, start_date, end_date)
        self.assertEqual(len(filtered_events), 0)

    def test_malformed_data(self):
        malformed_ics_content = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
SUMMARY:Malformed Event
DTSTART:not a date
LOCATION:Test Location
END:VEVENT
END:VCALENDAR"""

        malformed_path = Path(self.temp_dir.name) / "malformed.ics"
        malformed_path.write_text(malformed_ics_content)

        calendar = read_ics_file(malformed_path)
        events = extract_events(calendar)

        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0]["start"])
        self.assertEqual(events[0]["summary"], "Malformed Event")
        self.assertEqual(events[0]["location"], "Test Location")

        malformed_path.unlink()

    def test_get_events_between_dates_with_invalid_dates(self):
        calendar = read_ics_file(self.complex_ics_path)
        events = extract_events(calendar)

        invalid_event = {
            "summary": "Invalid Date Event",
            "start": "not a timestamp",
            "end": "also not a timestamp",
        }
        events.append(invalid_event)

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        filtered_events = get_events_between_dates(events, start_date, end_date)

        self.assertNotIn("Invalid Date Event", [e["summary"] for e in filtered_events])
        self.assertEqual(len(filtered_events), 5)  # Original number of valid events

    def test_parse_datetime_with_various_formats(self):
        calendar = read_ics_file(self.complex_ics_path)
        events = extract_events(calendar)

        malformed_ics_content = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
SUMMARY:Invalid Format Event
DTSTART:2024-13-45
END:VEVENT
BEGIN:VEVENT
SUMMARY:Empty Date Event
DTSTART:
DTEND:
END:VEVENT
BEGIN:VEVENT
SUMMARY:Invalid String Event
DTSTART:hello world
DTEND:goodbye world
END:VEVENT
BEGIN:VEVENT
SUMMARY:Invalid String Event
DTSTART:2024-12-25
DTEND:goodbye world
END:VEVENT
END:VCALENDAR"""

        malformed_path = Path(self.temp_dir.name) / "malformed_dates.ics"
        malformed_path.write_text(malformed_ics_content)

        calendar = read_ics_file(malformed_path)
        events = extract_events(calendar)

        self.assertEqual(len(events), 4)
        for event in events:
            self.assertIsNone(event["start"])
            self.assertIsNone(event["end"])

        malformed_path.unlink()


if __name__ == "__main__":
    unittest.main()
