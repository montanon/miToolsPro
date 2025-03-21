import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd
from icalendar import Calendar, Event

from mitoolspro.files.ics import (
    convert_to_dataframe,
    count_events_by_date,
    extract_events,
    format_event_for_display,
    get_events_between_dates,
    get_unique_attendees,
    get_unique_organizers,
    read_ics_file,
)


class TestICSFunctionality(unittest.TestCase):
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
        self.temp_dir = tempfile.mkdtemp()
        self.ics_path = Path(self.temp_dir) / "test.ics"
        self.ics_path.write_text(self.sample_ics_content)

    def tearDown(self):
        if self.ics_path.exists():
            self.ics_path.unlink()
        os.rmdir(self.temp_dir)

    def test_read_ics_file(self):
        calendar = read_ics_file(self.ics_path)
        self.assertIsInstance(calendar, Calendar)
        self.assertEqual(len(list(calendar.walk("VEVENT"))), 2)

        with self.assertRaises(FileNotFoundError):
            read_ics_file(Path(self.temp_dir) / "nonexistent.ics")

    def test_extract_events(self):
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

    def test_count_events_by_date(self):
        calendar = read_ics_file(self.ics_path)
        events = extract_events(calendar)
        event_counts = count_events_by_date(events)

        self.assertEqual(len(event_counts), 2)
        self.assertEqual(event_counts["2024-01-01"], 1)
        self.assertEqual(event_counts["2024-01-02"], 1)

    def test_get_unique_organizers(self):
        calendar = read_ics_file(self.ics_path)
        events = extract_events(calendar)
        organizers = get_unique_organizers(events)

        self.assertEqual(len(organizers), 2)
        self.assertIn("organizer1@example.com", organizers)
        self.assertIn("organizer2@example.com", organizers)

    def test_get_unique_attendees(self):
        calendar = read_ics_file(self.ics_path)
        events = extract_events(calendar)
        attendees = get_unique_attendees(events)

        self.assertEqual(len(attendees), 3)
        self.assertIn("attendee1@example.com", attendees)
        self.assertIn("attendee2@example.com", attendees)
        self.assertIn("attendee3@example.com", attendees)

    def test_convert_to_dataframe(self):
        calendar = read_ics_file(self.ics_path)
        events = extract_events(calendar)
        df = convert_to_dataframe(events)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["summary"], "Test Event 1")
        self.assertEqual(df.iloc[1]["summary"], "Test Event 2")

    def test_get_events_between_dates(self):
        calendar = read_ics_file(self.ics_path)
        events = extract_events(calendar)

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 1)
        filtered_events = get_events_between_dates(events, start_date, end_date)
        self.assertEqual(len(filtered_events), 1)
        self.assertEqual(filtered_events[0]["summary"], "Test Event 1")

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        filtered_events = get_events_between_dates(events, start_date, end_date)
        self.assertEqual(len(filtered_events), 2)

        start_date = datetime(2024, 1, 3)
        end_date = datetime(2024, 1, 4)
        filtered_events = get_events_between_dates(events, start_date, end_date)
        self.assertEqual(len(filtered_events), 0)

    def test_format_event_for_display(self):
        calendar = read_ics_file(self.ics_path)
        events = extract_events(calendar)
        formatted_event = format_event_for_display(events[0])

        self.assertIn("Summary: Test Event 1", formatted_event)
        self.assertIn("Description: Test Description 1", formatted_event)
        self.assertIn("Organizer: organizer1@example.com", formatted_event)
        self.assertIn("attendee1@example.com, attendee2@example.com", formatted_event)

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


if __name__ == "__main__":
    unittest.main()
