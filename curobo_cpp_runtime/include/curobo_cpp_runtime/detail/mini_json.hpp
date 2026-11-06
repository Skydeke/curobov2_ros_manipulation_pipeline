// SPDX-License-Identifier: Apache-2.0
// Minimal, dependency-free JSON reader for the generated native artifact
// metadata and for the bridge's small config objects.
//
// The package deliberately does not parse YAML: robots/*.yaml remains the
// single source of truth and the export step writes manifest.json. This reader
// only has to understand the JSON the node and the exporter already produce.
#ifndef CUROBO_CPP_RUNTIME__DETAIL__MINI_JSON_HPP_
#define CUROBO_CPP_RUNTIME__DETAIL__MINI_JSON_HPP_

#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace curobo_cpp_runtime
{
namespace json
{

class Value
{
public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Value() = default;
  explicit Value(bool value) : type_(Type::Bool), bool_(value) {}
  explicit Value(double value) : type_(Type::Number), number_(value) {}
  explicit Value(std::string value) : type_(Type::String), string_(std::move(value)) {}
  explicit Value(std::vector<Value> value) : type_(Type::Array), array_(std::move(value)) {}
  explicit Value(std::map<std::string, Value> value)
  : type_(Type::Object), object_(std::move(value)) {}

  Type type() const {return type_;}
  bool is_null() const {return type_ == Type::Null;}
  bool is_bool() const {return type_ == Type::Bool;}
  bool is_number() const {return type_ == Type::Number;}
  bool is_string() const {return type_ == Type::String;}
  bool is_array() const {return type_ == Type::Array;}
  bool is_object() const {return type_ == Type::Object;}

  bool as_bool(bool fallback = false) const
  {
    return type_ == Type::Bool ? bool_ : fallback;
  }

  double as_double(double fallback = 0.0) const
  {
    return type_ == Type::Number ? number_ : fallback;
  }

  const std::string & as_string() const
  {
    static const std::string empty;
    return type_ == Type::String ? string_ : empty;
  }

  std::string as_string(const std::string & fallback) const
  {
    return type_ == Type::String ? string_ : fallback;
  }

  const std::vector<Value> & as_array() const
  {
    static const std::vector<Value> empty;
    return type_ == Type::Array ? array_ : empty;
  }

  const std::map<std::string, Value> & as_object() const
  {
    static const std::map<std::string, Value> empty;
    return type_ == Type::Object ? object_ : empty;
  }

  const Value * find(const std::string & key) const
  {
    if (type_ != Type::Object) {
      return nullptr;
    }
    const auto it = object_.find(key);
    return it == object_.end() ? nullptr : &it->second;
  }

  const Value * find_path(const std::string & dotted) const
  {
    const Value * current = this;
    std::size_t start = 0;
    while (start < dotted.size()) {
      const std::size_t dot = dotted.find('.', start);
      const std::string key = dotted.substr(start, dot - start);
      current = current == nullptr ? nullptr : current->find(key);
      if (current == nullptr) {
        return nullptr;
      }
      if (dot == std::string::npos) {
        break;
      }
      start = dot + 1;
    }
    return current;
  }

  std::string string_at(
    const std::string & dotted, const std::string & fallback = "") const
  {
    const Value * value = find_path(dotted);
    return value != nullptr ? value->as_string(fallback) : fallback;
  }

  double number_at(const std::string & dotted, double fallback = 0.0) const
  {
    const Value * value = find_path(dotted);
    return value != nullptr ? value->as_double(fallback) : fallback;
  }

  bool bool_at(const std::string & dotted, bool fallback = false) const
  {
    const Value * value = find_path(dotted);
    return value != nullptr ? value->as_bool(fallback) : fallback;
  }

private:
  Type type_{Type::Null};
  bool bool_{false};
  double number_{0.0};
  std::string string_;
  std::vector<Value> array_;
  std::map<std::string, Value> object_;
};

namespace detail
{
class Parser
{
public:
  explicit Parser(const std::string & text) : text_(text) {}

  Value parse()
  {
    skip_ws();
    Value value = parse_value();
    skip_ws();
    if (pos_ != text_.size()) {
      fail("trailing characters");
    }
    return value;
  }

private:
  void skip_ws()
  {
    while (pos_ < text_.size() &&
      std::isspace(static_cast<unsigned char>(text_[pos_])) != 0)
    {
      ++pos_;
    }
  }

  [[noreturn]] void fail(const std::string & what) const
  {
    std::ostringstream out;
    out << "JSON parse error at byte " << pos_ << ": " << what;
    throw std::runtime_error(out.str());
  }

  bool consume(char expected)
  {
    if (pos_ < text_.size() && text_[pos_] == expected) {
      ++pos_;
      return true;
    }
    return false;
  }

  Value parse_value()
  {
    if (pos_ >= text_.size()) {
      fail("unexpected end of input");
    }
    switch (text_[pos_]) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': return Value(parse_string());
      case 't':
        expect_literal("true");
        return Value(true);
      case 'f':
        expect_literal("false");
        return Value(false);
      case 'n':
        expect_literal("null");
        return Value();
      default:
        if (text_[pos_] == '-' || std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
          return Value(parse_number());
        }
        fail("unexpected character");
    }
  }

  void expect_literal(const char * literal)
  {
    const std::size_t length = std::char_traits<char>::length(literal);
    if (text_.compare(pos_, length, literal) != 0) {
      fail("invalid literal");
    }
    pos_ += length;
  }

  static int hex_value(char c)
  {
    if (c >= '0' && c <= '9') {return c - '0';}
    if (c >= 'a' && c <= 'f') {return c - 'a' + 10;}
    if (c >= 'A' && c <= 'F') {return c - 'A' + 10;}
    return -1;
  }

  static void append_utf8(std::string & out, unsigned int codepoint)
  {
    if (codepoint <= 0x7F) {
      out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
      out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
      out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
      out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
  }

  std::string parse_string()
  {
    if (!consume('"')) {
      fail("expected string");
    }
    std::string out;
    while (pos_ < text_.size()) {
      const char c = text_[pos_++];
      if (c == '"') {
        return out;
      }
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (pos_ >= text_.size()) {
        fail("unterminated escape");
      }
      const char escape = text_[pos_++];
      switch (escape) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
            if (pos_ + 4 > text_.size()) {
              fail("short unicode escape");
            }
            unsigned int codepoint = 0;
            for (int i = 0; i < 4; ++i) {
              const int value = hex_value(text_[pos_++]);
              if (value < 0) {
                fail("invalid unicode escape");
              }
              codepoint = (codepoint << 4) | static_cast<unsigned int>(value);
            }
            append_utf8(out, codepoint);
            break;
          }
        default:
          fail("invalid escape");
      }
    }
    fail("unterminated string");
  }

  double parse_number()
  {
    const std::size_t start = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') {
      ++pos_;
    }
    while (pos_ < text_.size() &&
      std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0)
    {
      ++pos_;
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() &&
        std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0)
      {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
        ++pos_;
      }
      while (pos_ < text_.size() &&
        std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0)
      {
        ++pos_;
      }
    }
    const std::string token = text_.substr(start, pos_ - start);
    if (token.empty() || token == "-") {
      fail("invalid number");
    }
    char * end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end == nullptr || *end != '\0') {
      fail("invalid number");
    }
    return value;
  }

  Value parse_array()
  {
    consume('[');
    std::vector<Value> values;
    skip_ws();
    if (consume(']')) {
      return Value(std::move(values));
    }
    while (true) {
      skip_ws();
      values.push_back(parse_value());
      skip_ws();
      if (consume(']')) {
        return Value(std::move(values));
      }
      if (!consume(',')) {
        fail("expected ',' or ']'");
      }
    }
  }

  Value parse_object()
  {
    consume('{');
    std::map<std::string, Value> values;
    skip_ws();
    if (consume('}')) {
      return Value(std::move(values));
    }
    while (true) {
      skip_ws();
      if (pos_ >= text_.size() || text_[pos_] != '"') {
        fail("expected object key");
      }
      std::string key = parse_string();
      skip_ws();
      if (!consume(':')) {
        fail("expected ':'");
      }
      skip_ws();
      values.emplace(std::move(key), parse_value());
      skip_ws();
      if (consume('}')) {
        return Value(std::move(values));
      }
      if (!consume(',')) {
        fail("expected ',' or '}'");
      }
    }
  }

  const std::string & text_;
  std::size_t pos_{0};
};
}  // namespace detail

inline Value parse(const std::string & text)
{
  return detail::Parser(text).parse();
}

inline std::string escape(const std::string & value)
{
  std::string out;
  out.reserve(value.size() + 2);
  for (char c : value) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c;
    }
  }
  return out;
}

inline std::string quote(const std::string & value)
{
  return "\"" + escape(value) + "\"";
}

}  // namespace json
}  // namespace curobo_cpp_runtime

#endif  // CUROBO_CPP_RUNTIME__DETAIL__MINI_JSON_HPP_
