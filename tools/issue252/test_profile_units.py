"""Profiler API device durations are microseconds, never already seconds."""
from types import SimpleNamespace
import unittest


class ProfileUnitsTests(unittest.TestCase):
    def test_device_and_cpu_microseconds_are_both_converted(self):
        from tools.issue252.profile_units import operator_record
        event=SimpleNamespace(key='nccl:all_reduce',count=2,self_cpu_time_total=125000.,self_device_time_total=2500000.)
        self.assertEqual(operator_record(event),{'name':'nccl:all_reduce','calls':2,'self_cpu_seconds':.125,'self_device_seconds':2.5})


if __name__=='__main__':unittest.main()
