/* SPDX-License-Identifier: MIT */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"

struct Word {
    size_t begin;
    size_t length;
    size_t token;
    int64_t id;
};

using Pronunciations = std::vector<std::vector<std::string>>;
using Lexicon = std::unordered_map<int64_t, Pronunciations>;

static constexpr int64_t kUnknown = 100, kStart = 101, kEnd = 102, kMask = 103;
static constexpr size_t kVocabulary = 30522;
static constexpr size_t kMaximumTokens = 56;
static constexpr size_t kBatch = 6;

static std::vector<std::string> vowels(const std::vector<std::string>& phones) {
    std::vector<std::string> result;
    for (const auto& phone : phones) {
        if (phone.size() == 2 && std::string_view("AEIOU").find(phone[0]) != std::string_view::npos)
            result.push_back(phone);
    }
    return result;
}

static float phonetic_distance(const Pronunciations& first, const Pronunciations& second) {
    float best = 10;
    for (const auto& from : first) {
        for (const auto& to : second) {
            if (vowels(from) != vowels(to)) continue;
            std::vector<float> row(to.size() + 1);
            std::iota(row.begin(), row.end(), 0.0f);
            for (size_t i = 0; i < from.size(); ++i) {
                float diagonal = row[0]++;
                for (size_t j = 0; j < to.size(); ++j) {
                    const auto& a = from[i];
                    const auto& b = to[j];
                    const bool nearby = (a == "S" && b == "Z") || (a == "Z" && b == "S") ||
                                        (a == "P" && b == "F") || (a == "F" && b == "P");
                    const float distance = a == b ? 0.0f : nearby ? 0.5f : 1.0f;
                    const float above = row[j + 1];
                    row[j + 1] = std::min({above + 1, row[j] + 1, diagonal + distance});
                    diagonal = above;
                }
            }
            best = std::min(best, row.back());
        }
    }
    return best;
}

static void tokenize(const std::string& text, const std::unordered_map<std::string, int64_t>& dictionary,
                     const Lexicon& lexicon, std::vector<int64_t>& tokens, std::vector<Word>& eligible) {
    for (size_t begin = 0; begin < text.size();) {
        const auto ch = static_cast<unsigned char>(text[begin]);
        if (std::isspace(ch)) {
            ++begin;
            continue;
        }
        size_t end = begin + 1;
        const bool ordinary_word = std::isalnum(ch);
        if (ordinary_word) {
            while (end < text.size() && std::isalnum(static_cast<unsigned char>(text[end]))) ++end;
        }
        std::string word = text.substr(begin, end - begin);
        const bool changed_case = std::any_of(word.begin(), word.end(), [](unsigned char c) {
            return std::isupper(c);
        });
        for (auto& c : word) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        const auto direct = dictionary.find(word);
        if (direct != dictionary.end()) {
            const int64_t id = direct->second;
            const bool in_contraction = (begin > 0 && text[begin - 1] == '\'') ||
                                        (end < text.size() && text[end] == '\'');
            if (begin > 0 && ordinary_word && word.size() >= 3 && !changed_case && !in_contraction &&
                lexicon.contains(id)) {
                eligible.push_back({begin, end - begin, tokens.size(), id});
            }
            tokens.push_back(id);
        } else {
            std::vector<int64_t> pieces;
            for (size_t offset = 0; offset < word.size();) {
                size_t stop = word.size();
                for (; stop > offset; --stop) {
                    const auto piece = dictionary.find((offset ? "##" : "") + word.substr(offset, stop - offset));
                    if (piece != dictionary.end()) {
                        pieces.push_back(piece->second);
                        break;
                    }
                }
                if (stop == offset) {
                    pieces = {kUnknown};
                    break;
                }
                offset = stop;
            }
            tokens.insert(tokens.end(), pieces.begin(), pieces.end());
        }
        begin = end;
    }
}

static bool protected_word(const std::string& word) {
    static const std::vector<std::string> protected_terms = {
        "no", "not", "never", "without", "only", "except",
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "twenty", "hundred", "thousand",
        "delete", "remove", "send", "move", "cancel", "buy", "stop", "start",
    };
    return std::find(protected_terms.begin(), protected_terms.end(), word) != protected_terms.end();
}

static std::string refine(const std::string& text, const std::string& previous,
                          const std::filesystem::path& assets) {
    if (text.empty() || text.size() > 300 ||
        std::any_of(text.begin(), text.end(), [](unsigned char c) { return c > 127; }))
        return text;
    std::ifstream word_file(assets / "context-vocab.txt");
    std::unordered_map<std::string, int64_t> dictionary;
    std::vector<std::string> vocabulary;
    for (std::string line; std::getline(word_file, line);) {
        dictionary.emplace(line, vocabulary.size());
        vocabulary.push_back(std::move(line));
    }
    if (vocabulary.size() != kVocabulary) return text;

    std::ifstream phoneme_file(assets / "context-phones.txt");
    Lexicon lexicon;
    for (std::string line; std::getline(phoneme_file, line);) {
        std::istringstream row(line);
        int64_t id;
        if (!(row >> id)) continue;
        std::vector<std::string> phones;
        for (std::string p; row >> p;) phones.push_back(std::move(p));
        if (!phones.empty()) lexicon[id].push_back(std::move(phones));
    }

    std::vector<int64_t> ids{kStart};
    std::vector<Word> eligible;
    tokenize(text, dictionary, lexicon, ids, eligible);
    ids.push_back(kEnd);
    if (eligible.empty() || ids.size() > kMaximumTokens) return text;
    std::vector<int64_t> context;
    std::vector<Word> ignored;
    if (previous.size() <= 300) tokenize(previous, dictionary, lexicon, context, ignored);
    if (!context.empty() && context.size() + ids.size() < kMaximumTokens) {
        ids.insert(ids.end(), context.begin(), context.end());
        ids.push_back(kEnd);
    }
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "speech-context");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    options.DisableCpuMemArena();
    Ort::Session model(env, (assets / "context-model.onnx").c_str(), options);
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    constexpr float minimum_score = 2.30f, minimum_margin = 0.75f;
    float best = 0.0f, second = 0.0f;
    std::string replacement;
    const Word* replace = nullptr;
    std::vector<size_t> order(kVocabulary);
    for (size_t start = 0; start < eligible.size(); start += kBatch) {
        const size_t batch = std::min(kBatch, eligible.size() - start);
        const size_t length = ids.size();
        std::vector<int64_t> input(batch * length), attention(batch * length, 1);
        for (size_t i = 0; i < batch; ++i) {
            std::copy(ids.begin(), ids.end(), input.begin() + i * length);
            input[i * length + eligible[start + i].token] = kMask;
        }
        const int64_t shape[] = {static_cast<int64_t>(batch), static_cast<int64_t>(length)};
        Ort::Value tensors[] = {
            Ort::Value::CreateTensor<int64_t>(memory, input.data(), input.size(), shape, 2),
            Ort::Value::CreateTensor<int64_t>(memory, attention.data(), attention.size(), shape, 2),
        };
        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {"logits"};
        auto output = model.Run(Ort::RunOptions{}, input_names, tensors, 2, output_names, 1);
        const auto dimensions = output[0].GetTensorTypeAndShapeInfo().GetShape();
        if (dimensions != std::vector<int64_t>{shape[0], shape[1], kVocabulary}) return text;
        const float* logits = output[0].GetTensorData<float>();
        for (size_t i = 0; i < batch; ++i) {
            const Word& word = eligible[start + i];
            const std::string& original = vocabulary[word.id];
            if (protected_word(original)) continue;
            const float* scores = logits + (i * length + word.token) * kVocabulary;
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + 80, order.end(),
                              [&](size_t a, size_t b) { return scores[a] > scores[b]; });
            for (size_t n = 0; n < 80; ++n) {
                const size_t candidate = order[n];
                const float gain = scores[candidate] - scores[word.id];
                if (gain < 4.1f) break;
                const auto phones = lexicon.find(candidate);
                const auto& alternative = vocabulary[candidate];
                if (phones == lexicon.end() || protected_word(alternative)) continue;
                if (gain < 7.0f && original.ends_with('s') != alternative.ends_with('s')) continue;
                const float distance = phonetic_distance(lexicon.at(word.id), phones->second);
                if (distance > 3.0f) continue;
                const float score = gain - 0.9f * distance;
                if (score > best) {
                    second = best;
                    best = score;
                    replacement = alternative;
                    replace = &word;
                } else {
                    second = std::max(second, score);
                }
            }
        }
    }
    if (!replace || best < minimum_score || best - second < minimum_margin) return text;
    return text.substr(0, replace->begin) + replacement + text.substr(replace->begin + replace->length);
}

int main() {
    std::string text, previous;
    if (!std::getline(std::cin, text)) return 1;
    std::getline(std::cin, previous);
    try {
        std::cout << refine(text, previous, std::filesystem::canonical("/proc/self/exe").parent_path()) << '\n';
    } catch (const std::exception& error) {
        std::cerr << "speech context: " << error.what() << '\n';
        return 1;
    }
}
