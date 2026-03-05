import asyncio
import json
from datetime import datetime, timedelta
from supabase import create_client


DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

TOOL_DEFINITIONS = [
    {
        "name": "lookup_caller",
        "description": (
            "Look up an existing caller by their phone number. "
            "Always call this first when a caller wants to book an appointment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "phone_number": {
                    "type": "string",
                    "description": "Caller's phone number in E.164 format e.g. +14155550101",
                }
            },
            "required": ["phone_number"],
        },
    },
    {
        "name": "create_caller",
        "description": "Create a new caller record when the caller is not found in the database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "first_name":   {"type": "string"},
                "last_name":    {"type": "string"},
                "phone_number": {"type": "string"},
            },
            "required": ["first_name", "last_name", "phone_number"],
        },
    },
    {
        "name": "get_pets",
        "description": "Get all pets registered under a caller.",
        "input_schema": {
            "type": "object",
            "properties": {
                "caller_id": {"type": "string", "description": "The caller's UUID"},
            },
            "required": ["caller_id"],
        },
    },
    {
        "name": "create_pet",
        "description": "Register a new pet for a caller.",
        "input_schema": {
            "type": "object",
            "properties": {
                "caller_id":   {"type": "string"},
                "pet_name":    {"type": "string"},
                "pet_species": {
                    "type": "string",
                    "enum": ["dog", "cat", "rabbit", "bird", "reptile", "small_mammal", "other"],
                },
                "pet_breed": {"type": "string"},
            },
            "required": ["caller_id", "pet_name", "pet_species"],
        },
    },
    {
        "name": "get_doctors",
        "description": "Get the list of available doctors, optionally filtered by specialty.",
        "input_schema": {
            "type": "object",
            "properties": {
                "specialty": {
                    "type": "string",
                    "description": (
                        "Optional specialty filter e.g. 'General Practice', "
                        "'Surgery', 'Dermatology', 'Dentistry'"
                    ),
                }
            },
        },
    },
    {
        "name": "get_available_slots",
        "description": "Get available appointment time slots for a doctor on a specific date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "doctor_id": {"type": "string"},
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format e.g. 2026-03-10",
                },
            },
            "required": ["doctor_id", "date"],
        },
    },
    {
        "name": "book_appointment",
        "description": (
            "Book a confirmed appointment. "
            "Only call this after verbally confirming all details with the caller."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "caller_id":        {"type": "string"},
                "pet_id":           {"type": "string"},
                "doctor_id":        {"type": "string"},
                "reason_for_visit": {"type": "string"},
                "appointment_date": {
                    "type": "string",
                    "description": "ISO datetime string e.g. 2026-03-10T10:00:00",
                },
            },
            "required": ["caller_id", "pet_id", "doctor_id", "reason_for_visit", "appointment_date"],
        },
    },
]


class BookingTools:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase = create_client(supabase_url, supabase_key)

    # ── Tool implementations (sync, run via asyncio.to_thread) ────────────────

    def _lookup_caller(self, phone_number: str) -> dict:
        result = self.supabase.table("caller").select("*").eq("phone_number", phone_number).execute()
        if result.data:
            return {"found": True, "caller": result.data[0]}
        return {"found": False}

    def _create_caller(self, first_name: str, last_name: str, phone_number: str) -> dict:
        result = self.supabase.table("caller").insert({
            "first_name":   first_name,
            "last_name":    last_name,
            "phone_number": phone_number,
        }).execute()
        return {"caller": result.data[0]}

    def _get_pets(self, caller_id: str) -> dict:
        result = self.supabase.table("pet").select("*").eq("caller_id", caller_id).execute()
        return {"pets": result.data}

    def _create_pet(self, caller_id: str, pet_name: str, pet_species: str, pet_breed: str = None) -> dict:
        data = {"caller_id": caller_id, "pet_name": pet_name, "pet_species": pet_species}
        if pet_breed:
            data["pet_breed"] = pet_breed
        result = self.supabase.table("pet").insert(data).execute()
        return {"pet": result.data[0]}

    def _get_doctors(self, specialty: str = None) -> dict:
        query = self.supabase.table("doctor").select("*")
        if specialty:
            query = query.ilike("specialty", f"%{specialty}%")
        result = query.execute()
        return {"doctors": result.data}

    def _get_available_slots(self, doctor_id: str, date: str) -> dict:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD."}

        day_of_week = target_date.weekday()  # 0=Monday

        sched = (
            self.supabase.table("doctor_schedule")
            .select("*")
            .eq("doctor_id", doctor_id)
            .eq("day_of_week", day_of_week)
            .eq("is_active", True)
            .execute()
        )

        if not sched.data:
            return {
                "available": False,
                "message": f"Doctor is not available on {DAY_NAMES[day_of_week]}s.",
            }

        schedule  = sched.data[0]
        slot_min  = schedule["slot_duration_minutes"]
        start_t   = datetime.strptime(schedule["start_time"][:5], "%H:%M").time()
        end_t     = datetime.strptime(schedule["end_time"][:5],   "%H:%M").time()
        start     = datetime.combine(target_date, start_t)
        end       = datetime.combine(target_date, end_t)

        all_slots = []
        cur = start
        while cur + timedelta(minutes=slot_min) <= end:
            all_slots.append(cur)
            cur += timedelta(minutes=slot_min)

        # Fetch already-booked appointments that day
        booked = (
            self.supabase.table("appointment")
            .select("appointment_date")
            .eq("doctor_id", doctor_id)
            .gte("appointment_date", f"{date}T00:00:00")
            .lte("appointment_date", f"{date}T23:59:59")
            .neq("status", "cancelled")
            .execute()
        )

        booked_times = set()
        for appt in booked.data:
            raw = appt["appointment_date"]
            dt  = datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
            booked_times.add(dt)

        available = [s.strftime("%H:%M") for s in all_slots if s not in booked_times]
        return {
            "available": True,
            "date":      str(target_date),
            "day":       DAY_NAMES[day_of_week],
            "slots":     available,
        }

    def _book_appointment(
        self,
        caller_id: str,
        pet_id: str,
        doctor_id: str,
        reason_for_visit: str,
        appointment_date: str,
    ) -> dict:
        # Create a call record first (required by FK on appointment.call_id)
        call_result = self.supabase.table("call").insert({
            "caller_id": caller_id,
            "pet_id":    pet_id,
            "intent":    "appointment_booking",
        }).execute()
        call_id = call_result.data[0]["call_id"]

        appt = self.supabase.table("appointment").insert({
            "call_id":          call_id,
            "pet_id":           pet_id,
            "doctor_id":        doctor_id,
            "reason_for_visit": reason_for_visit,
            "appointment_date": appointment_date,
            "status":           "pending",
        }).execute()
        return {"success": True, "appointment": appt.data[0]}

    # ── Async dispatcher ──────────────────────────────────────────────────────

    async def execute(self, tool_name: str, tool_input: dict) -> str:
        fn_map = {
            "lookup_caller":      self._lookup_caller,
            "create_caller":      self._create_caller,
            "get_pets":           self._get_pets,
            "create_pet":         self._create_pet,
            "get_doctors":        self._get_doctors,
            "get_available_slots": self._get_available_slots,
            "book_appointment":   self._book_appointment,
        }
        fn = fn_map.get(tool_name)
        if fn is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = await asyncio.to_thread(fn, **tool_input)
            return json.dumps(result, default=str)
        except Exception as e:
            print(f"[Tool] Error in {tool_name}: {e}")
            return json.dumps({"error": str(e)})
