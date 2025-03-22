import pickle
import unittest
from unittest import TestCase

from mitoolspro.utils.objects import (
    AttrDict,
    LazyDict,
    LazyList,
    StringMapper,
    _new_attr_dict_,
)


class TestStringMapper(TestCase):
    def setUp(self):
        self.relations = {"pretty1": "ugly1", "pretty2": "ugly2"}
        self.mapper = StringMapper(self.relations)

    def test_add_relation(self):
        self.mapper.add_relation("pretty3", "ugly3")
        self.assertEqual(self.mapper.prettify_str("ugly3"), "pretty3")
        self.assertEqual(self.mapper.uglify_str("pretty3"), "ugly3")

    def test_add_relation_case_insensitive(self):
        mapper = StringMapper(self.relations, case_sensitive=False)
        mapper.add_relation("Pretty4", "Ugly4")
        self.assertEqual(mapper.prettify_str("ugly4"), "pretty4")
        self.assertEqual(mapper.uglify_str("pretty4"), "ugly4")
        self.assertEqual(mapper.prettify_str("Ugly4"), "pretty4")
        self.assertEqual(mapper.uglify_str("Pretty4"), "ugly4")

    def test_add_relation_case_sensitive(self):
        mapper = StringMapper(self.relations, case_sensitive=True)
        mapper.add_relation("Pretty4", "Ugly4")
        self.assertEqual(mapper.prettify_str("Ugly4"), "Pretty4")
        self.assertEqual(mapper.uglify_str("Pretty4"), "Ugly4")
        self.assertNotEqual(mapper.prettify_str("Ugly4"), "pretty4")
        self.assertNotEqual(mapper.uglify_str("Pretty4"), "ugly4")

    def test_add_relation_duplicate(self):
        with self.assertRaises(ValueError):
            self.mapper.add_relation("pretty1", "ugly3")

    def test_prettify_str(self):
        self.assertEqual(self.mapper.prettify_str("ugly1"), "pretty1")

    def test_prettify_str_not_found(self):
        with self.assertRaises(ValueError):
            self.mapper.prettify_str("ugly3")

    def test_prettify_strs(self):
        self.assertEqual(
            self.mapper.prettify_strs(["ugly1", "ugly2"]), ["pretty1", "pretty2"]
        )

    def test_uglify_str(self):
        self.assertEqual(self.mapper.uglify_str("pretty1"), "ugly1")

    def test_uglify_str_not_found(self):
        with self.assertRaises(ValueError):
            self.mapper.uglify_str("pretty3")

    def test_uglify_strs(self):
        self.assertEqual(
            self.mapper.uglify_strs(["pretty1", "pretty2"]), ["ugly1", "ugly2"]
        )

    def test_remap_str(self):
        self.assertEqual(self.mapper.remap_str("pretty1"), "ugly1")
        self.assertEqual(self.mapper.remap_str("ugly1"), "pretty1")

    def test_remap_str_not_found(self):
        with self.assertRaises(ValueError):
            self.mapper.remap_str("pretty3")

    def test_remap_strs(self):
        self.assertEqual(
            self.mapper.remap_strs(["pretty1", "pretty2"]), ["ugly1", "ugly2"]
        )
        self.assertEqual(
            self.mapper.remap_strs(["ugly1", "ugly2"]), ["pretty1", "pretty2"]
        )

    def test_remap_strs_mixed(self):
        with self.assertRaises(ValueError):
            self.mapper.remap_strs(["pretty1", "ugly2"])

    def test_is_pretty(self):
        self.assertTrue(self.mapper.is_pretty("pretty1"))
        self.assertFalse(self.mapper.is_pretty("ugly1"))

    def test_is_ugly(self):
        self.assertTrue(self.mapper.is_ugly("ugly1"))
        self.assertFalse(self.mapper.is_ugly("pretty1"))

    def test_case_insensitive(self):
        mapper = StringMapper(self.relations, case_sensitive=False)
        self.assertTrue(mapper.is_pretty("PRETTY1"))
        self.assertTrue(mapper.is_ugly("UGLY1"))
        self.assertEqual(mapper.prettify_str("UGLY1"), "pretty1")
        self.assertEqual(mapper.uglify_str("PRETTY1"), "ugly1")

    def test_prettify_str_pass_if_mapped(self):
        mapper = StringMapper(self.relations, pass_if_mapped=True)
        self.assertEqual(mapper.prettify_str("pretty1"), "pretty1")

    def test_prettify_str_pass_if_mapped_not_found(self):
        mapper = StringMapper(self.relations, pass_if_mapped=True)
        with self.assertRaises(ValueError):
            mapper.prettify_str("ugly3")

    def test_uglify_str_pass_if_mapped(self):
        mapper = StringMapper(self.relations, pass_if_mapped=True)
        self.assertEqual(mapper.uglify_str("ugly1"), "ugly1")

    def test_uglify_str_pass_if_mapped_not_found(self):
        mapper = StringMapper(self.relations, pass_if_mapped=True)
        with self.assertRaises(ValueError):
            mapper.uglify_str("pretty3")

    def test_remap_str_pass_if_mapped(self):
        mapper = StringMapper(self.relations, pass_if_mapped=True)
        self.assertEqual(mapper.remap_str("pretty1"), "ugly1")
        self.assertEqual(mapper.remap_str("ugly1"), "pretty1")

    def test_remap_str_pass_if_mapped_not_found(self):
        mapper = StringMapper(self.relations, pass_if_mapped=True)
        with self.assertRaises(ValueError):
            mapper.remap_str("pretty3")

    def test_case_insensitive_pass_if_mapped(self):
        mapper = StringMapper(self.relations, case_sensitive=False, pass_if_mapped=True)
        self.assertEqual(mapper.prettify_str("PRETTY1"), "pretty1")
        self.assertEqual(mapper.uglify_str("UGLY1"), "ugly1")
        self.assertEqual(mapper.prettify_str("pretty1"), "pretty1")
        self.assertEqual(mapper.uglify_str("ugly1"), "ugly1")

    def test_case_insensitive_pass_if_mapped_not_found(self):
        mapper = StringMapper(self.relations, case_sensitive=False, pass_if_mapped=True)
        with self.assertRaises(ValueError):
            mapper.prettify_str("UGLY3")
        with self.assertRaises(ValueError):
            mapper.uglify_str("PRETTY3")

    def test_case_insensitive_pass_if_mapped_remap(self):
        mapper = StringMapper(self.relations, case_sensitive=False, pass_if_mapped=True)
        self.assertEqual(mapper.remap_str("PRETTY1"), "ugly1")
        self.assertEqual(mapper.remap_str("UGLY1"), "pretty1")
        self.assertEqual(mapper.remap_str("pretty1"), "ugly1")
        self.assertEqual(mapper.remap_str("ugly1"), "pretty1")

    def test_case_insensitive_pass_if_mapped_remap_not_found(self):
        mapper = StringMapper(self.relations, case_sensitive=False, pass_if_mapped=True)
        with self.assertRaises(ValueError):
            mapper.remap_str("PRETTY3")
        with self.assertRaises(ValueError):
            mapper.remap_str("ugly3")


class TestAttrDict(TestCase):
    def test_init_empty(self):
        ad = AttrDict()
        self.assertEqual(len(ad), 0)
        self.assertEqual(str(ad), "AttrDict{}")

    def test_init_with_dict(self):
        input_dict = {"a": 1, "b": 2}
        ad = AttrDict(input_dict)
        self.assertEqual(len(ad), 2)
        self.assertEqual(ad["a"], 1)
        self.assertEqual(ad["b"], 2)

    def test_init_with_kwargs(self):
        ad = AttrDict(x=10, y=20)
        self.assertEqual(ad.x, 10)
        self.assertEqual(ad.y, 20)

    def test_init_with_sequence_of_tuples(self):
        ad = AttrDict([("key1", "val1"), ("key2", "val2")])
        self.assertEqual(ad.key1, "val1")
        self.assertEqual(ad.key2, "val2")

    def test_attribute_access(self):
        ad = AttrDict({"foo": "bar"})
        self.assertEqual(ad.foo, "bar")
        self.assertEqual(ad["foo"], "bar")
        ad.foo = "baz"
        self.assertEqual(ad.foo, "baz")
        self.assertEqual(ad["foo"], "baz")
        ad["foo"] = "qux"
        self.assertEqual(ad.foo, "qux")
        self.assertEqual(ad["foo"], "qux")

    def test_attribute_error(self):
        ad = AttrDict({"a": 1})
        with self.assertRaises(AttributeError):
            _ = ad.non_existent

    def test_item_access(self):
        ad = AttrDict({"one": 1, "two": 2})
        self.assertEqual(ad["one"], 1)
        self.assertEqual(ad["two"], 2)
        with self.assertRaises(KeyError):
            _ = ad["three"]

    def test_item_assignment(self):
        ad = AttrDict()
        ad["test"] = 123
        self.assertEqual(ad.test, 123)

    def test_item_deletion(self):
        ad = AttrDict({"a": 1, "b": 2})
        del ad["a"]
        self.assertNotIn("a", ad)
        with self.assertRaises(KeyError):
            del ad["a"]

    def test_attribute_deletion(self):
        ad = AttrDict({"x": 10})
        del ad.x
        self.assertNotIn("x", ad)
        with self.assertRaises(KeyError):
            del ad.x

    def test_private_dict_protection(self):
        ad = AttrDict()
        with self.assertRaises(KeyError):
            ad["__private_dict__"] = {}
        with self.assertRaises(AttributeError):
            ad.__private_dict__ = {}

    def test_len(self):
        ad = AttrDict({"a": 1, "b": 2, "c": 3})
        self.assertEqual(len(ad), 3)
        del ad["b"]
        self.assertEqual(len(ad), 2)

    def test_contains(self):
        ad = AttrDict(a=1, b=2)
        self.assertIn("a", ad)
        self.assertNotIn("c", ad)

    def test_keys_values_items(self):
        ad = AttrDict({"a": 1, "b": 2})
        self.assertEqual(set(ad.keys()), {"a", "b"})
        self.assertEqual(set(ad.values()), {1, 2})
        self.assertEqual(set(ad.items()), {("a", 1), ("b", 2)})

    def test_iter(self):
        ad = AttrDict({"x": 100, "y": 200})
        keys = list(iter(ad))
        self.assertIn("x", keys)
        self.assertIn("y", keys)

    def test_update_with_dict(self):
        ad = AttrDict({"a": 1})
        ad.update({"b": 2})
        self.assertEqual(ad.b, 2)
        self.assertEqual(len(ad), 2)

    def test_update_with_kwargs(self):
        ad = AttrDict({"a": 1})
        ad.update(c=3)
        self.assertEqual(ad.c, 3)

    def test_update_with_iterable(self):
        ad = AttrDict({"a": 1})
        ad.update([("b", 2), ("c", 3)])
        self.assertEqual(ad.b, 2)
        self.assertEqual(ad.c, 3)

    def test_clear(self):
        ad = AttrDict(a=1, b=2)
        ad.clear()
        self.assertEqual(len(ad), 0)
        self.assertNotIn("a", ad)

    def test_copy(self):
        ad = AttrDict(a=1, b=2)
        ad_copy = ad.copy()
        self.assertIsNot(ad, ad_copy)
        self.assertEqual(ad_copy.a, 1)
        self.assertEqual(ad_copy.b, 2)
        ad_copy.a = 100
        self.assertEqual(ad.a, 1)
        self.assertEqual(ad_copy.a, 100)

    def test_pop(self):
        ad = AttrDict(a=1, b=2)
        val = ad.pop("a")
        self.assertEqual(val, 1)
        self.assertNotIn("a", ad)

        default_val = ad.pop("missing", "default")
        self.assertEqual(default_val, "default")

    def test_reduce(self):
        ad = AttrDict(a=1, b=2)
        func, args = ad.__reduce__()
        self.assertEqual(func, _new_attr_dict_)
        self.assertEqual(set(args), {("a", 1), ("b", 2)})

    def test_pickle(self):
        ad = AttrDict(x=42, y="foo")
        data = pickle.dumps(ad)
        loaded = pickle.loads(data)
        self.assertEqual(loaded.x, 42)
        self.assertEqual(loaded.y, "foo")
        self.assertIsInstance(loaded, AttrDict)

    def test_repr_str(self):
        ad = AttrDict(a=1, b=2)
        rep = repr(ad)
        self.assertTrue(rep.startswith("AttrDict{"))
        self.assertIn("'a': 1", rep)
        self.assertIn("'b': 2", rep)
        self.assertTrue(rep.endswith("}"))
        self.assertEqual(str(ad), rep)

    def test_dir(self):
        ad = AttrDict(a=1, b=2, non_id_key="val", _private=3)
        directory = dir(ad)
        self.assertIn("a", directory)
        self.assertIn("b", directory)
        self.assertIn("non_id_key", directory)
        self.assertIn("_private", directory)

    def test_setattr(self):
        ad = AttrDict()
        ad.foo = "bar"
        self.assertEqual(ad["foo"], "bar")

    def test_delattr(self):
        ad = AttrDict(foo="bar", baz="qux")
        del ad.foo
        self.assertNotIn("foo", ad)
        with self.assertRaises(KeyError):
            del ad.foo

    def test_invalid_setattr_private_dict(self):
        ad = AttrDict()
        with self.assertRaises(AttributeError):
            ad.__private_dict__ = {}

    def test_reserved_key_error(self):
        ad = AttrDict()
        with self.assertRaises(KeyError):
            ad["__private_dict__"] = 123


class TestLazyList(TestCase):
    def setUp(self):
        class TestableLazyList(LazyList):
            def __init__(self):
                super().__init__()
                self.load_called = False

            def load(self):
                self.load_called = True
                list.extend(self, [1, 2, 3])

        self.TestableLazyList = TestableLazyList

    def test_no_load_before_usage(self):
        d = self.TestableLazyList()
        self.assertFalse(d.load_called)
        repr_str = repr(d)
        self.assertTrue(d.load_called)
        self.assertIn("1", repr_str)
        self.assertEqual(len(d), 3)

    def test_len_triggers_load(self):
        d = self.TestableLazyList()
        self.assertFalse(d.load_called)
        length = len(d)
        self.assertTrue(d.load_called)
        self.assertEqual(length, 3)

    def test_iter_triggers_load(self):
        d = self.TestableLazyList()
        self.assertFalse(d.load_called)
        items = list(d)  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertEqual(items, [1, 2, 3])

    def test_contains_triggers_load(self):
        d = self.TestableLazyList()
        self.assertFalse(d.load_called)
        self.assertIn(1, d)  # Should trigger load
        self.assertTrue(d.load_called)

    def test_insert_triggers_load(self):
        d = self.TestableLazyList()
        self.assertFalse(d.load_called)
        d.insert(0, 0)  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertEqual(d[0], 0)
        self.assertEqual(d[1], 1)

    def test_append_triggers_load(self):
        d = self.TestableLazyList()
        self.assertFalse(d.load_called)
        d.append(4)  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertEqual(d[-1], 4)
        self.assertEqual(d[0], 1)

    def test_extend_triggers_load(self):
        d = self.TestableLazyList()
        self.assertFalse(d.load_called)
        d.extend([4, 5])  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertEqual(d[-2:], [4, 5])
        self.assertEqual(d[:3], [1, 2, 3])

    def test_remove_triggers_load(self):
        d = self.TestableLazyList()
        self.assertFalse(d.load_called)
        d.remove(2)  # Should trigger load and remove '2'
        self.assertTrue(d.load_called)
        self.assertNotIn(2, d)
        self.assertEqual(d, [1, 3])

    def test_remove_non_existent(self):
        d = self.TestableLazyList()
        _ = len(d)
        with self.assertRaises(ValueError):
            d.remove(99)

    def test_pop_triggers_load(self):
        d = self.TestableLazyList()
        self.assertFalse(d.load_called)
        val = d.pop()  # Should trigger load, default pop removes last
        self.assertTrue(d.load_called)
        self.assertEqual(val, 3)
        self.assertEqual(d, [1, 2])

    def test_pop_with_index(self):
        d = self.TestableLazyList()
        _ = len(d)
        val = d.pop(0)
        self.assertEqual(val, 1)
        self.assertEqual(d, [2, 3])

    def test_pop_non_existent(self):
        d = self.TestableLazyList()
        # Trigger load first
        _ = len(d)
        with self.assertRaises(IndexError):
            d.pop(99)

    def test_multiple_instances(self):
        d1 = self.TestableLazyList()
        d2 = self.TestableLazyList()
        self.assertFalse(d1.load_called)
        self.assertFalse(d2.load_called)
        _ = len(d1)  # trigger load in d1
        self.assertTrue(d1.load_called)
        self.assertFalse(d2.load_called)
        _ = len(d2)  # now trigger load in d2
        self.assertTrue(d2.load_called)

    def test_repr_after_load(self):
        d = self.TestableLazyList()
        _ = len(d)  # trigger load
        rep = repr(d)
        self.assertIn("1", rep)
        self.assertIn("2", rep)
        self.assertIn("3", rep)

    def test_after_load_normal_list_behavior(self):
        d = self.TestableLazyList()
        _ = len(d)  # trigger load
        d.append(10)
        d.insert(0, 0)
        d.remove(3)
        val = d.pop()
        self.assertEqual(val, 10)
        self.assertEqual(d, [0, 1, 2])
        items = list(d)
        self.assertEqual(items, [0, 1, 2])

    def test_subclass_behavior(self):
        class CustomLoadLazyList(LazyList):
            def load(self):
                list.append(self, 100)

        d = CustomLoadLazyList()
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0], 100)

    def test_contains_non_existent_after_load(self):
        d = self.TestableLazyList()
        _ = len(d)  # trigger load
        self.assertNotIn(999, d)

    def test_extend_merge_values(self):
        d = self.TestableLazyList()
        _ = len(d)  # trigger load
        d.extend([4, 5, 6])
        self.assertEqual(d, [1, 2, 3, 4, 5, 6])

    def test_insert_positions(self):
        d = self.TestableLazyList()
        _ = len(d)  # trigger load
        d.insert(0, 0)  # front
        d.insert(2, "x")  # middle
        d.insert(len(d), "end")  # at the end
        self.assertEqual(d, [0, 1, "x", 2, 3, "end"])


class TestLazyDict(TestCase):
    def setUp(self):
        class TestableLazyDict(LazyDict):
            def __init__(self):
                super().__init__()
                self.load_called = False

            def load(self):
                self.load_called = True
                dict_values = {"a": 1, "b": 2, "c": 3}
                for key, value in dict_values.items():
                    dict.__setitem__(self, key, value)

        self.TestableLazyDict = TestableLazyDict

    def test_no_load_before_usage(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        repr_str = repr(d)
        self.assertTrue(d.load_called)
        self.assertIn("a", repr_str)
        self.assertEqual(len(d), 3)

    def test_len_triggers_load(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        length = len(d)  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertEqual(length, 3)

    def test_iter_triggers_load(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        keys = list(d)  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertCountEqual(keys, ["a", "b", "c"])

    def test_contains_triggers_load(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        self.assertIn("a", d)  # Should trigger load
        self.assertTrue(d.load_called)

    def test_getitem_triggers_load(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        value = d["a"]  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertEqual(value, 1)

    def test_setitem_triggers_load(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        d["d"] = 4  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertEqual(d["d"], 4)
        self.assertEqual(d["a"], 1)

    def test_get_triggers_load(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        val = d.get("a")  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertEqual(val, 1)

    def test_setdefault_triggers_load(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        d.setdefault("e", 5)  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertEqual(d["e"], 5)
        val = d.setdefault("a", 10)
        self.assertEqual(val, 1)
        self.assertEqual(d["a"], 1)

    def test_items_triggers_load(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        it = d.items()  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertCountEqual(it, [("a", 1), ("b", 2), ("c", 3)])

    def test_keys_triggers_load(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        k = d.keys()  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertCountEqual(k, ["a", "b", "c"])

    def test_values_triggers_load(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        v = d.values()  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertCountEqual(list(v), [1, 2, 3])

    def test_update_triggers_load(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        d.update({"f": 6})  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertEqual(d["f"], 6)
        self.assertEqual(d["a"], 1)

    def test_pop_triggers_load(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        val = d.pop("a", None)  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertEqual(val, 1)
        self.assertNotIn("a", d)

    def test_popitem_triggers_load(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        key, val = d.popitem()  # Should trigger load
        self.assertTrue(d.load_called)
        self.assertIn(key, ["a", "b", "c"])
        self.assertIn(val, [1, 2, 3])
        self.assertNotIn(key, d)

    def test_no_double_load(self):
        d = self.TestableLazyDict()
        self.assertFalse(d.load_called)
        _ = d["a"]  # first trigger
        self.assertTrue(d.load_called)
        # Reset the flag to see if it's called again
        d.load_called = False
        _ = d["b"]  # access another key, should NOT reload
        self.assertFalse(d.load_called)

    def test_multiple_instances(self):
        d1 = self.TestableLazyDict()
        d2 = self.TestableLazyDict()
        self.assertFalse(d1.load_called)
        self.assertFalse(d2.load_called)
        _ = d1["a"]
        self.assertTrue(d1.load_called)
        self.assertFalse(d2.load_called)
        _ = d2["b"]
        self.assertTrue(d2.load_called)

    def test_non_existent_key_after_load(self):
        d = self.TestableLazyDict()
        _ = d["a"]  # trigger load
        with self.assertRaises(KeyError):
            _ = d["not_here"]
        val = d.get("not_here")
        self.assertIsNone(val)

    def test_pop_non_existent_key(self):
        d = self.TestableLazyDict()
        _ = d["a"]  # trigger load
        val = d.pop("not_here", "default")
        self.assertEqual(val, "default")
        with self.assertRaises(KeyError):
            d.pop("still_not_here")

    def test_repr_after_load(self):
        d = self.TestableLazyDict()
        _ = d["a"]  # trigger load
        rep = repr(d)
        self.assertIn("a", rep)
        self.assertIn("b", rep)
        self.assertIn("c", rep)

    def test_update_merge_values(self):
        d = self.TestableLazyDict()
        _ = d["a"]  # trigger load
        d.update({"a": 10, "x": 99})
        self.assertEqual(d["a"], 10)
        self.assertEqual(d["x"], 99)
        self.assertEqual(d["b"], 2)

    def test_setdefault_existing_key(self):
        d = self.TestableLazyDict()
        _ = d["a"]  # trigger load
        original = d.setdefault("a", 999)
        self.assertEqual(original, 1)
        self.assertEqual(d["a"], 1)

    def test_subclass_behavior(self):
        class CustomLoadLazyDict(LazyDict):
            def load(self):
                dict.__setitem__(self, "z", 100)

        d = CustomLoadLazyDict()
        _ = d["z"]
        self.assertEqual(d["z"], 100)

    def test_load_once_then_normal_dict(self):
        d = self.TestableLazyDict()
        _ = d["a"]  # trigger load
        d["g"] = 7
        self.assertEqual(d["g"], 7)
        val = d.pop("b")
        self.assertEqual(val, 2)
        self.assertNotIn("b", d)
        keys = list(d.keys())
        self.assertCountEqual(keys, ["a", "c", "g"])
        items = list(d.items())
        self.assertIn(("a", 1), items)
        self.assertIn(("c", 3), items)
        self.assertIn(("g", 7), items)


if __name__ == "__main__":
    unittest.main()
