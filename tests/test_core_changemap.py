""" Test ChangeMap Module """

from bananas.changemap.changemap import ChangeMap
from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class TestUtils(ProfilingTestCase):
    def test_change_map_build(self):

        # Delete the first column
        change_map = ChangeMap(3, idx_del=[0])
        change_map_dict = change_map.build()
        self.assertEqual({0: -1, 1: 0, 2: 1}, change_map_dict)

        # Delete the first column and re-add it
        change_map = ChangeMap(3, idx_del=[0], idx_add=[0])
        change_map_dict = change_map.build()
        self.assertEqual({-1: [0], 0: -1, 1: 1, 2: 2}, change_map_dict)


main()
