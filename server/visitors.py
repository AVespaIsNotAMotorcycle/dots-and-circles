'''
Counts each day's unique visitors without keeping anything about them:
each is known only by a hash of their IP and browser with a random salt
made fresh each day and never written anywhere but the server's own /tmp,
which is deleted the next day (and on every restart). Only the counts
leave the server, printed to the log (Heroku's logs, on Heroku):

  visitors 2026-11-13: 5 (3 read something)

A line is printed whenever either count grows, so the last one of a day
is its total. Days are in UTC. The gunicorn workers share the day's file,
locked while it's read and written, so a visitor whose requests go to
both counts once. A restart starts the day's counts again.
'''

import datetime
import fcntl
import glob
import hashlib
import os
import secrets

PREFIX = "dots-and-circles-visitors-"
# Hex digits of each visitor's hash kept: plenty to tell a day's visitors
# apart
HASH_LENGTH = 16


class VisitorCounter:
    def __init__(self, folder, today=None):
        self.folder = folder
        # A function giving today's date, for testing
        self.today = today or (lambda: datetime.datetime.now(
            datetime.timezone.utc).date())

    def path(self, day):
        return os.path.join(self.folder, f"{PREFIX}{day.isoformat()}.txt")

    def record(self, ip, browser, read):
        '''
        Counts a request from ip with the User-Agent browser; read is
        whether it read something. Returns the day's (visitors, readers),
        and prints them if either grew.
        '''
        day = self.today()
        path = self.path(day)
        for old in glob.glob(os.path.join(self.folder, PREFIX + "*.txt")):
            if old != path:
                try:
                    os.remove(old)
                except FileNotFoundError:
                    pass  # Another worker got there first
        with open(path, "a+", encoding="ascii") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            try:
                file.seek(0)
                lines = file.read().split()
                if not lines:
                    lines = [secrets.token_hex(32)]
                    file.write(lines[0] + "\n")
                salt = lines[0]
                visitors = set(line[2:] for line in lines[1:]
                               if line.startswith("v:"))
                readers = set(line[2:] for line in lines[1:]
                              if line.startswith("r:"))
                key = hashlib.sha256(f"{salt}|{ip}|{browser}".encode(
                    "utf-8")).hexdigest()[:HASH_LENGTH]
                grew = False
                if key not in visitors:
                    visitors.add(key)
                    file.write(f"v:{key}\n")
                    grew = True
                if read and key not in readers:
                    readers.add(key)
                    file.write(f"r:{key}\n")
                    grew = True
                file.flush()
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)
        if grew:
            print(f"visitors {day.isoformat()}: {len(visitors)} "
                  f"({len(readers)} read something)", flush=True)
        return len(visitors), len(readers)
