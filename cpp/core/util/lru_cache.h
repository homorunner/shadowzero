#pragma once

#include <list>
#include <map>
#include <optional>

// from boost::compute::detail::lru_cache
// a cache which evicts the least recently used item when it is full
template <class Key, class Value>
class lru_cache {
 public:
  typedef std::list<Key> List;
  typedef std::map<Key, std::pair<Value, typename List::iterator>> Map;

  lru_cache(size_t capacity) : m_capacity(capacity) {}

  ~lru_cache() {}

  size_t size() const { return m_map.size(); }

  size_t capacity() const { return m_capacity; }

  bool empty() const { return m_map.empty(); }

  bool contains(const Key& key) { return m_map.find(key) != m_map.end(); }

  void insert(const Key& key, const Value& value) {
    typename Map::iterator i = m_map.find(key);
    if (i == m_map.end()) {
      // insert item into the cache, but first check if it is full
      if (size() >= m_capacity) {
        // cache is full, evict the least recently used item
        evict();
      }

      // insert the new item
      m_list.push_front(key);
      m_map[key] = std::make_pair(value, m_list.begin());
    }
  }

  Value must_get(const Key& key) {
    // lookup value in the cache
    auto i = m_map.find(key);

    // return the value, but first update its place in the most
    // recently used list
    auto j = i->second.second;
    if (j != m_list.begin()) {
      // move item to the front of the most recently used list
      m_list.erase(j);
      m_list.push_front(key);

      // update iterator in map
      j = m_list.begin();
      const Value& value = i->second.first;
      m_map[key] = std::make_pair(value, j);

      // return the value
      return value;
    } else {
      // the item is already at the front of the most recently
      // used list so just return it
      return i->second.first;
    }
  }

  void clear() {
    m_map.clear();
    m_list.clear();
  }

 private:
  void evict() {
    // evict item from the end of most recently used list
    auto i = --m_list.end();
    m_map.erase(*i);
    m_list.erase(i);
  }

  Map m_map;
  List m_list;
  size_t m_capacity;
};
