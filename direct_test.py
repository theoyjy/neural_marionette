#!/usr/bin/env python3
"""
Direct test of the timeout mechanism.
"""

print("Testing improved timeout mechanism...")

try:
    from GenerateSkel import run_dembone_with_timeout
    print("✅ Successfully imported run_dembone_with_timeout")
    
    # Test the heartbeat mechanism
    import time
    import threading
    
    def dummy_compute():
        print("🧪 Starting dummy compute test...")
        time.sleep(15)  # Sleep for 15 seconds to test heartbeat
        print("🧪 Dummy compute completed")
    
    class MockDemBones:
        def compute(self):
            dummy_compute()
    
    mock_dem = MockDemBones()
    
    print("🧪 Testing timeout with 10s limit (should timeout)...")
    success, error = run_dembone_with_timeout(mock_dem, timeout_seconds=10)
    
    if not success and "Timeout" in str(error):
        print("✅ Timeout mechanism working correctly!")
        print(f"✅ Error message: {error}")
    else:
        print(f"❌ Timeout didn't work as expected: success={success}, error={error}")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("🏁 Direct test completed")
