#pragma once

#include <rapidjson/document.h>
#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ov {
namespace genai {

class LLMStyler {
public:
  /**
   * @brief Construct a new LLMStyler object
   * @param style_name: Style name.
   * @param system_prompt: System role prompt.
   * @param roles: Roles name. [`system_name`, `user_name`, `assistant_name`]
   * @param support_func_call: If support function call.
   * @param func_call_observation_words: Function call observation words, used to early stop.
   * @param add_generation_prompt: If true, will add generation prompt in the end of prompt.
   */
  LLMStyler(std::string style_name,
            std::string system_prompt,
            const std::vector<std::string>& roles,
            bool support_func_call,
            std::string func_call_observation_words,
            bool add_generation_prompt = false)
      : style_name_(std::move(style_name))
      , system_prompt_(std::move(system_prompt))
      , roles_(roles)
      , support_func_call_(support_func_call)
      , func_call_observation_words_(std::move(func_call_observation_words))
      , add_generation_prompt_(add_generation_prompt) {}
  virtual ~LLMStyler() = default;

  /**
   * Apply chat template.
   * @param chat_template: chat_template in tokenizer_config.json.
   */
  virtual void ApplyChatTemplate(const std::string& chat_template) { chat_template_ = chat_template; }

  /**
   * @brief Build prompt for model input from OpenAI interface json body request.
   * @param json_body: Json body from client.
   * @return <if_function_call, prompt, img_urls>: if_function_call is true if the prompt contains function call.
   */
  virtual std::tuple<bool, std::string, std::vector<std::string>> BuildPrompt(const nlohmann::json& json_body);

  /**
   * @brief Parse function call response from generated text and build content and tool_calls array of message
   * member of OpenAI interface response.
   * @param gen_txt: Generated text.
   * @param req_id: Request id.
   * @param message: Message member of OpenAI interface response format.
   * @param allocator: Json allocator.
   * @return stop reason.
   */
  virtual std::string ParseFunctionCall(const std::string& gen_txt,
                                        int64_t req_id,
                                        rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                        rapidjson::MemoryPoolAllocator<>& allocator);

  const std::string& GetRole(const std::string& role_name) {
    if (role_name == "system") {
      return roles_[0];
    } else if (role_name == "user") {
      return roles_[1];
    } else if (role_name == "assistant") {
      return roles_[2];
    } else {
      return role_name;
    }
  }

  [[nodiscard]] const std::string& style_name() const { return style_name_; }
  [[nodiscard]] const std::string& system_prompt() const { return system_prompt_; }
  [[nodiscard]] const std::vector<std::string>& roles() const { return roles_; }
  [[nodiscard]] bool support_func_call() const { return support_func_call_; }
  [[nodiscard]] const std::string& func_call_observation_words() const { return func_call_observation_words_; }
  [[nodiscard]] bool add_generation_prompt() const { return add_generation_prompt_; }

protected:
  std::string style_name_;                  // Style name.
  std::string system_prompt_;               // System role prompt.
  std::vector<std::string> roles_;          // Roles name. [`system_name`, `user_name`, `assistant_name`]
  bool support_func_call_ = false;          // If support function call.
  std::string func_call_observation_words_; // Function call observation words. Used to early stop.
  bool add_generation_prompt_ = false;      // If true, will add generation prompt in the end of prompt.
  std::string chat_template_;               // chat_template in tokenizer_config.json.
};

class Qwen3Styler : public LLMStyler {
public:
  Qwen3Styler() : LLMStyler("qwen3", "", {"system", "user", "assistant"}, true, "", true) {
    tool_prompt_pre_ =
      "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function "
      "signatures within <tools></tools> XML tags:\n<tools>";
    tool_prompt_post_ =
      "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call>"
      "</tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
      "</tool_call>";
  }
  Qwen3Styler(std::string style_name,
              std::string system_prompt,
              const std::vector<std::string>& roles,
              bool support_func_call,
              std::string func_call_observation_words,
              bool add_generation_prompt = false,
              std::string tool_prompt_pre = "",
              std::string tool_prompt_post = "")
      : LLMStyler(std::move(style_name),
                  std::move(system_prompt),
                  roles,
                  support_func_call,
                  std::move(func_call_observation_words),
                  add_generation_prompt) {
    tool_prompt_pre_ = std::move(tool_prompt_pre);
    tool_prompt_post_ = std::move(tool_prompt_post);
  }
  ~Qwen3Styler() override = default;

  /**
   * @brief Build prompt for model input from OpenAI interface json body request.
   * @param json_body: Json body from client.
   * @return <if_function_call, prompt, img_urls>: if_function_call is true if the prompt contains function call.
   */
  std::tuple<bool, std::string, std::vector<std::string>> BuildPrompt(const nlohmann::json& json_body) override;

  /**
   * @brief Parse function call response from generated text and build content and tool_calls array of message
   * member of OpenAI interface response.
   * @param gen_txt: Generated text.
   * @param req_id: Request id.
   * @param message: Message member of OpenAI interface response format.
   * @param allocator: Json allocator.
   * @return stop reason.
   */
  std::string ParseFunctionCall(const std::string& gen_txt,
                                int64_t req_id,
                                rapidjson::GenericValue<rapidjson::UTF8<>>& message,
                                rapidjson::MemoryPoolAllocator<>& allocator) override;

protected:
  std::string tool_prompt_pre_;
  std::string tool_prompt_post_;
};

}
}