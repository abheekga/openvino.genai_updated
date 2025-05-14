#include "llm_styler.hpp"
#include "utils.hpp"

#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <regex>
#include <unordered_set>
#include <vector>

namespace ov {
namespace genai {

std::tuple<bool, std::string, std::vector<std::string>> LLMStyler::BuildPrompt(const nlohmann::json& json_body) {
  throw std::runtime_error("BuildPrompt not implemented.");
}

std::string LLMStyler::ParseFunctionCall(const std::string& gen_txt,
                                         int64_t req_id,
                                         rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                         rapidjson::MemoryPoolAllocator<>& allocator) {
  throw std::runtime_error("ParseFunctionCall not implemented.");
}

std::tuple<bool, std::string, std::vector<std::string>> Qwen3Styler::BuildPrompt(const nlohmann::json& json_body) {
  std::string prompt;

  if (!json_body.contains("messages") || !json_body["messages"].is_array()) {
    throw std::invalid_argument("`messages` not found or not an array");
  }
  if (json_body["messages"].empty()) {
    throw std::invalid_argument("`messages` is empty");
  }

  bool func_call = false;
  bool skip_first = false;
  if (json_body.contains("tools") && json_body["tools"].is_array()) {
    // Parse tools.
    prompt += "<|im_start|>system\n";
    if (json_body["messages"][0].contains("role") && json_body["messages"][0]["role"].is_string() &&
        std::string(json_body["messages"][0]["role"].get<std::string>()) == "system") {
      if (json_body["messages"][0].contains("content") && json_body["messages"][0]["content"].is_string()) {
        prompt += json_body["messages"][0]["content"].get<std::string>();
      }
      prompt += "\n\n";
      skip_first = true;
    }

    prompt += tool_prompt_pre_;
    for (auto& tool : json_body["tools"]) { // parse per tool to json.
      prompt += "\n";
      // rapidjson::StringBuffer buffer;
      // rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      // tool.accept(writer);
      // // Add space after colon and comma to match python json.
      // prompt += ov::genai::utils::JsonAddSpaceAfterColonAndComma(buffer.get<std::string>());
      std::string json_str = tool.dump();
      prompt += ov::genai::utils::JsonAddSpaceAfterColonAndComma(json_str);
    }
    prompt += tool_prompt_post_;

    prompt += "<|im_end|>\n";

    func_call = true;
  } else if (json_body["messages"][0].contains("role") && json_body["messages"][0]["role"].is_string() &&
             std::string(json_body["messages"][0]["role"].get<std::string>()) == "system") {
    // Parse system message.
    prompt += "<|im_start|>system\n";
    if (json_body["messages"][0].contains("content") && json_body["messages"][0]["content"].is_string()) {
      prompt += json_body["messages"][0]["content"].get<std::string>();
    }
    prompt += "<|im_end|>\n";
    skip_first = true;
  }

  bool multi_step_tool = true;
  size_t last_query_index = json_body["messages"].size() - 1; // last user query index
  for (int i = last_query_index; i >= 0; i--) {
    if (!json_body["messages"][i].contains("role") || !json_body["messages"][i]["role"].is_string()) {
      throw std::invalid_argument("`role` not found or not a string");
    }

    auto role = std::string(json_body["messages"][i]["role"].get<std::string>());
    std::string content;
    if (json_body["messages"][i].contains("content") && json_body["messages"][i]["content"].is_string()) {
      content = json_body["messages"][i]["content"].get<std::string>();
    }
    if (multi_step_tool && role == "user" && !ov::genai::utils::StartsWith(content, "<tool_response>") &&
        !ov::genai::utils::EndsWith(content, "</tool_response>")) {
      multi_step_tool = false;
      last_query_index = i;
    }
  }

  for (size_t i = 0; i < json_body["messages"].size(); i++) {
    if (skip_first) {
      skip_first = false;
      continue;
    }
    auto& message = json_body["messages"][i];

    if (!message.contains("role") || !message["role"].is_string()) {
      throw std::invalid_argument("`role` not found or not a string");
    }
    auto role = std::string(message["role"].get<std::string>());

    if (role == "user" || role == "system") {
      std::string content;
      if (message.contains("content") && message["content"].is_string()) {
        content = json_body["messages"][i]["content"].get<std::string>();
      }

      prompt += "<|im_start|>";
      prompt += GetRole(role);
      prompt += "\n";
      prompt += message["content"].get<std::string>();
      prompt += "<|im_end|>\n";
    } else if (role == "assistant") {
      std::string content;
      if (message.contains("content") && message["content"].is_string()) {
        content = json_body["messages"][i]["content"].get<std::string>();
      }
      std::string reasoning_content;

      if (content.find("</think>") != std::string::npos) {
        // {%- set reasoning_content =
        // message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
        reasoning_content = content.substr(0, content.find("</think>"));
        reasoning_content = ov::genai::utils::Rstrip(reasoning_content, "\n");
        reasoning_content = reasoning_content.substr(reasoning_content.find("<think>") + 7);
        reasoning_content = ov::genai::utils::Lstrip(reasoning_content, "\n");

        // {%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}
        content = content.substr(content.find("</think>") + 8);
        content = ov::genai::utils::Lstrip(content, "\n");
      }

      if (i > last_query_index) {
        if (i == json_body["messages"].size() - 1 ||
            (i != json_body["messages"].size() - 1 && !reasoning_content.empty())) {
          prompt += "<|im_start|>";
          prompt += GetRole(role);
          prompt += "\n<think>\n";
          prompt += reasoning_content;
          prompt += "\n</think>\n\n";
          prompt += content;
        } else {
          prompt += "<|im_start|>";
          prompt += GetRole(role);
          prompt += "\n";
          prompt += content;
        }
      } else {
        prompt += "<|im_start|>";
        prompt += GetRole(role);
        prompt += "\n";
        prompt += content;
      }

      if (message.contains("tool_calls") && message["tool_calls"].is_array()) {
        for (size_t j = 0; j < message["tool_calls"].size(); j++) {
          if ((j == 0 && !content.empty()) || j != 0) {
            prompt += "\n";
          }
          auto& tool_call = message["tool_calls"][j];
          if (tool_call.contains("function") && tool_call["function"].is_object()) {
            auto& function = tool_call["function"];
            prompt += "<tool_call>\n{\"name\": \"";
            if (function.contains("name") && function["name"].is_string()) {
              prompt += function["name"].get<std::string>();
            } else {
              throw std::invalid_argument("`name` not found in `function` or not a string");
            }
            prompt += "\", \"arguments\": ";
            if (function.contains("arguments")) {
              if (function["arguments"].is_string()) {
                prompt += function["arguments"].get<std::string>();
              } else if (function["arguments"].is_object()) {
                // rapidjson::StringBuffer buffer;
                // rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                // function["arguments"].accept(writer);
                // prompt += ov::genai::utils::JsonAddSpaceAfterColonAndComma(buffer.get<std::string>());
                std::string json_str = function["arguments"].dump();
                prompt += ov::genai::utils::JsonAddSpaceAfterColonAndComma(json_str);
              } else {
                throw std::invalid_argument("`arguments` not found in `function` or not a string");
              }
            } else {
              throw std::invalid_argument("`arguments` not found in `function`");
            }
            prompt += "}\n</tool_call>";
          } else {
            throw std::invalid_argument("`function` not found in `tool_call` or not an object");
          }
        }
      }
      prompt += "<|im_end|>\n";
    } else if (role == "tool") {
      if (message.contains("content") && message["content"].is_string()) {
        if (i == 0 || std::string(json_body["messages"][i - 1]["role"].get<std::string>()) != "tool") {
          prompt += "<|im_start|>user";
        }
        prompt += "\n<tool_response>\n";
        prompt += message["content"].get<std::string>();
        prompt += "\n</tool_response>";
        if (i == json_body["messages"].size() - 1 ||
            std::string(json_body["messages"][i + 1]["role"].get<std::string>()) != "tool") {
          prompt += "<|im_end|>\n";
        }
      }
      if (i == json_body["messages"].size() - 1) {
        func_call = false;
      }
    } else {
      throw std::invalid_argument("Unsupported role: " + role);
    }
  }

  if (add_generation_prompt()) {
    prompt += "<|im_start|>";
    prompt += GetRole("assistant");
    prompt += "\n";
  }

  if (json_body.contains("enable_thinking") && json_body["enable_thinking"].is_boolean() &&
      !json_body["enable_thinking"].get<bool>()) {
    prompt += "<think>\n\n</think>\n\n";
  }

  return {func_call, prompt, {}};
}

std::string Qwen3Styler::ParseFunctionCall(const std::string& gen_txt,
                                           int64_t req_id,
                                           rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                           rapidjson::MemoryPoolAllocator<>& allocator) {
  return "tool_calls";
}
}
}