import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import pandas as pd
from icalendar import Calendar, Event
from pandas import DataFrame


def read_ics_file(filepath: Union[str, Path]) -> Calendar:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ICS file not found: {filepath}")
    with filepath.open("r", encoding="utf-8") as file:
        return Calendar.from_ical(file.read())


def _parse_datetime(component: Event, field: str) -> Optional[pd.Timestamp]:
    try:
        if component.get(field):
            return pd.to_datetime(component.decoded(field))
    except Exception:
        pass
    return None


def _parse_attendees(attendees_field: Union[List[str], str]) -> List[str]:
    if not attendees_field:
        return []
    if isinstance(attendees_field, list):
        return [str(a).replace("mailto:", "") for a in attendees_field]
    return [str(attendees_field).replace("mailto:", "")]


def extract_events(
    cal: Calendar,
) -> List[Dict[str, Union[str, List[str], pd.Timestamp, None]]]:
    events = []
    for component in cal.walk(name="VEVENT"):
        events.append(
            {
                "summary": str(component.get("SUMMARY", "")),
                "description": str(component.get("DESCRIPTION", "")),
                "start": _parse_datetime(component, "DTSTART"),
                "end": _parse_datetime(component, "DTEND"),
                "organizer": str(component.get("ORGANIZER", "")).replace("mailto:", ""),
                "attendees": _parse_attendees(component.get("ATTENDEE")),
                "url": str(component.get("URL", "")),
                "uid": str(component.get("UID", "")),
                "transp": str(component.get("TRANSP", "")),
                "status": str(component.get("STATUS", "")),
                "sequence": component.get("SEQUENCE", ""),
                "rrule": component.get("RRULE", ""),
                "recurrence_id": str(component.get("RECURRENCE-ID", "")),
                "location": str(component.get("LOCATION", "")),
                "last_modified": _parse_datetime(component, "LAST-MODIFIED"),
                "exdate": component.get("EXDATE", ""),
                "dtstamp": _parse_datetime(component, "DTSTAMP"),
                "created": _parse_datetime(component, "CREATED"),
                "class": str(component.get("CLASS", "")),
                "attach": component.get("ATTACH", ""),
            }
        )
    return events


def count_events_by_date(
    events: List[Dict[str, Union[str, List[str], pd.Timestamp, None]]],
) -> Dict[str, int]:
    event_count = {}
    for event in events:
        if isinstance(event["start"], pd.Timestamp):
            date_str = event["start"].date().isoformat()
            event_count[date_str] = event_count.get(date_str, 0) + 1
    return event_count


def get_unique_organizers(
    events: List[Dict[str, Union[str, List[str], pd.Timestamp, None]]],
) -> Set[str]:
    return {str(event["organizer"]) for event in events if event.get("organizer")}


def get_unique_attendees(
    events: List[Dict[str, Union[str, List[str], pd.Timestamp, None]]],
) -> Set[str]:
    attendees = set()
    for event in events:
        attendees.update(event.get("attendees", []))
    return attendees


def convert_to_dataframe(
    events: List[Dict[str, Union[str, List[str], pd.Timestamp, None]]],
) -> DataFrame:
    return DataFrame(events)


def get_events_between_dates(
    events: List[Dict[str, Union[str, List[str], pd.Timestamp, None]]],
    start_date: datetime,
    end_date: datetime,
) -> List[Dict[str, Union[str, List[str], pd.Timestamp, None]]]:
    filtered_events = []
    for event in events:
        event_start = event.get("start")
        event_end = event.get("end")

        if not isinstance(event_start, pd.Timestamp):
            continue

        if not isinstance(event_end, pd.Timestamp):
            event_end = event_start

        event_start_date = event_start.date()
        event_end_date = event_end.date()
        start_date_obj = start_date.date()
        end_date_obj = end_date.date()

        if event_start_date <= end_date_obj and event_end_date >= start_date_obj:
            filtered_events.append(event)

    return filtered_events


def format_event_for_display(
    event: Dict[str, Union[str, List[str], pd.Timestamp, None]],
) -> str:
    return (
        f"Summary: {event.get('summary', '')}\n"
        f"Description: {event.get('description', '')}\n"
        f"Start: {event.get('start', '')}\n"
        f"End: {event.get('end', '')}\n"
        f"Location: {event.get('location', '')}\n"
        f"Organizer: {event.get('organizer', '')}\n"
        f"Attendees: {', '.join(event.get('attendees', []))}\n"
    )


if __name__ == "__main__":
    filepath = "example.ics"
    if os.path.exists(filepath):
        calendar = read_ics_file(filepath)
        events = extract_events(calendar)
        event_count_by_date = count_events_by_date(events)
        for date, count in event_count_by_date.items():
            print(f"{date}: {count} events")
        df = convert_to_dataframe(events)
        print(df)
        organizers = get_unique_organizers(events)
        print(f"Unique organizers: {organizers}")
        attendees = get_unique_attendees(events)
        print(f"Unique attendees: {attendees}")
        if events:
            print(format_event_for_display(events[0]))
    else:
        print(f"File {filepath} not found.")
