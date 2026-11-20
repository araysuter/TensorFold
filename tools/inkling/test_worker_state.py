import unittest

try:
    from worker import NoHistory
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Requires the pinned Inkling MLX environment')
class NoHistoryTests(unittest.TestCase):
    def test_finished_response_releases_its_cache_list(self):
        state = NoHistory()
        entry = object()
        live_cache = [entry]
        finished_cache = live_cache[:]
        state.insert_cache('model', [1, 2], finished_cache)
        self.assertEqual(finished_cache, [])
        self.assertEqual(live_cache, [entry])
        self.assertEqual(len(state), 0)
        self.assertEqual(state.nbytes, 0)
        self.assertEqual(state.fetch_nearest_cache('model', [1, 2]), (None, [1, 2]))


if __name__ == '__main__':
    unittest.main()
