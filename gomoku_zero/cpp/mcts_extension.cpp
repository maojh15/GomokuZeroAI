#include <torch/extension.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

struct Node {
    double prior = 0.0;
    int parent = -1;
    int move = -1;
    int visits = 0;
    double value_sum = 0.0;
    std::vector<int> children;
};

struct PendingRequest {
    int node_id = -1;
    int current_player = 0;
    std::vector<int> path;
    std::vector<int8_t> board;
};

class CppMCTSGame {
public:
    CppMCTSGame(
        int board_height,
        int board_width,
        int first_player,
        int second_player,
        int n_playout,
        double c_puct,
        int candidate_distance,
        bool tactical_shortcuts)
        : height_(board_height),
          width_(board_width),
          board_size_(board_height * board_width),
          first_player_(first_player),
          second_player_(second_player),
          n_playout_(n_playout),
          c_puct_(c_puct),
          candidate_distance_(candidate_distance),
          tactical_shortcuts_(tactical_shortcuts) {
        if (height_ <= 0 || width_ <= 0) {
            throw std::invalid_argument("board dimensions must be positive");
        }
        reset();
    }

    void reset() {
        board_.assign(board_size_, 0);
        current_player_ = first_player_;
        nodes_.clear();
        nodes_.push_back(Node{1.0, -1, -1, 0, 0.0, {}});
        root_ = 0;
        pending_.clear();
        next_request_id_ = 1;
    }

    py::dict request_leaf() {
        for (int i = 0; i < n_playout_; ++i) {
            py::dict request = request_one_leaf();
            const std::string status = py::cast<std::string>(request["status"]);
            if (status == "leaf") {
                return request;
            }
        }
        py::dict done;
        done["status"] = "done";
        return done;
    }

    void apply_evaluation(int request_id, const std::vector<float>& policy_probs, double value) {
        auto it = pending_.find(request_id);
        if (it == pending_.end()) {
            throw std::invalid_argument("unknown request_id");
        }
        PendingRequest pending = std::move(it->second);
        pending_.erase(it);

        std::vector<int> legal = candidate_moves(pending.board);
        double prob_sum = 0.0;
        for (int move : legal) {
            if (move >= 0 && move < static_cast<int>(policy_probs.size())) {
                double p = static_cast<double>(policy_probs[move]);
                if (std::isfinite(p) && p > 0.0) {
                    prob_sum += p;
                }
            }
        }
        const double uniform = legal.empty() ? 0.0 : 1.0 / static_cast<double>(legal.size());
        for (int move : legal) {
            double prior = uniform;
            if (prob_sum > 0.0 && move >= 0 && move < static_cast<int>(policy_probs.size())) {
                prior = std::max(0.0, static_cast<double>(policy_probs[move])) / prob_sum;
            }
            Node child;
            child.prior = prior;
            child.parent = pending.node_id;
            child.move = move;
            nodes_.push_back(std::move(child));
            nodes_[pending.node_id].children.push_back(static_cast<int>(nodes_.size()) - 1);
        }

        double leaf_value = value * 2.0 - 1.0;
        backup(pending.path, leaf_value);
    }

    py::tuple action_probs(double temp) const {
        const Node& root = nodes_[root_];
        std::vector<int> moves;
        std::vector<float> probs;
        if (root.children.empty()) {
            return py::make_tuple(moves, probs);
        }

        moves.reserve(root.children.size());
        std::vector<double> visits;
        visits.reserve(root.children.size());
        for (int child_id : root.children) {
            const Node& child = nodes_[child_id];
            moves.push_back(child.move);
            visits.push_back(static_cast<double>(child.visits));
        }

        probs.assign(moves.size(), 0.0f);
        if (temp <= 1e-3) {
            auto best_it = std::max_element(visits.begin(), visits.end());
            probs[static_cast<size_t>(std::distance(visits.begin(), best_it))] = 1.0f;
        } else {
            double sum = 0.0;
            for (double& visit : visits) {
                visit = std::pow(visit, 1.0 / temp);
                sum += visit;
            }
            if (sum <= 0.0 || !std::isfinite(sum)) {
                const float uniform = 1.0f / static_cast<float>(moves.size());
                std::fill(probs.begin(), probs.end(), uniform);
            } else {
                for (size_t i = 0; i < visits.size(); ++i) {
                    probs[i] = static_cast<float>(visits[i] / sum);
                }
            }
        }
        return py::make_tuple(moves, probs);
    }

    int tactical_move() const {
        if (!tactical_shortcuts_) {
            return -1;
        }
        std::vector<int> legal = legal_moves(board_);
        int winning = find_winning_move(board_, current_player_, legal);
        if (winning >= 0) {
            return winning;
        }
        return find_winning_move(board_, opponent(current_player_), legal);
    }

    void apply_move(int move) {
        if (move < 0 || move >= board_size_ || board_[move] != 0) {
            throw std::invalid_argument("illegal move");
        }
        board_[move] = static_cast<int8_t>(current_player_);

        int new_root = -1;
        for (int child_id : nodes_[root_].children) {
            if (nodes_[child_id].move == move) {
                new_root = child_id;
                break;
            }
        }
        if (new_root >= 0) {
            root_ = new_root;
            nodes_[root_].parent = -1;
        } else {
            nodes_.clear();
            nodes_.push_back(Node{1.0, -1, -1, 0, 0.0, {}});
            root_ = 0;
        }
        current_player_ = opponent(current_player_);
        pending_.clear();
    }

    py::tuple game_end_after_move(int move, int player) const {
        if (has_five_from(board_, move, player)) {
            return py::make_tuple(true, player);
        }
        for (int8_t cell : board_) {
            if (cell == 0) {
                return py::make_tuple(false, 0);
            }
        }
        return py::make_tuple(true, 0);
    }

    std::vector<int8_t> board() const { return board_; }
    int current_player() const { return current_player_; }
    int root_visits() const { return nodes_[root_].visits; }
    int n_playout() const { return n_playout_; }

private:
    py::dict request_one_leaf() {
        int node_id = root_;
        int player = current_player_;
        std::vector<int8_t> board = board_;
        std::vector<int> path{root_};
        int last_move = -1;
        int last_player = 0;

        while (!nodes_[node_id].children.empty()) {
            int child_id = select_child(node_id);
            int move = nodes_[child_id].move;
            board[move] = static_cast<int8_t>(player);
            last_move = move;
            last_player = player;
            player = opponent(player);
            node_id = child_id;
            path.push_back(node_id);
        }

        if (last_move >= 0) {
            if (has_five_from(board, last_move, last_player)) {
                double value = last_player == player ? 1.0 : -1.0;
                backup(path, value);
                py::dict terminal;
                terminal["status"] = "terminal";
                return terminal;
            }
            if (std::none_of(board.begin(), board.end(), [](int8_t v) { return v == 0; })) {
                backup(path, 0.0);
                py::dict terminal;
                terminal["status"] = "terminal";
                return terminal;
            }
        }

        int request_id = next_request_id_++;
        pending_[request_id] = PendingRequest{node_id, player, path, board};
        py::dict leaf;
        leaf["status"] = "leaf";
        leaf["request_id"] = request_id;
        leaf["state"] = encode_state(board, player);
        return leaf;
    }

    int select_child(int node_id) const {
        const Node& node = nodes_[node_id];
        double best_score = -std::numeric_limits<double>::infinity();
        int best_child = -1;
        double parent_visits_sqrt = std::sqrt(std::max(1, node.visits));
        for (int child_id : node.children) {
            const Node& child = nodes_[child_id];
            double q = child.visits == 0 ? 0.0 : child.value_sum / static_cast<double>(child.visits);
            double exploration = c_puct_ * child.prior * parent_visits_sqrt / (1.0 + child.visits);
            double score = -q + exploration;
            if (score > best_score) {
                best_score = score;
                best_child = child_id;
            }
        }
        if (best_child < 0) {
            throw std::runtime_error("node has no selectable child");
        }
        return best_child;
    }

    void backup(const std::vector<int>& path, double leaf_value) {
        double value = leaf_value;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            Node& node = nodes_[*it];
            node.visits += 1;
            node.value_sum += value;
            value = -value;
        }
    }

    std::vector<float> encode_state(const std::vector<int8_t>& board, int player) const {
        int other = opponent(player);
        std::vector<float> state(static_cast<size_t>(2 * board_size_), 0.0f);
        for (int i = 0; i < board_size_; ++i) {
            state[static_cast<size_t>(i)] = board[i] == player ? 1.0f : 0.0f;
            state[static_cast<size_t>(board_size_ + i)] = board[i] == other ? 1.0f : 0.0f;
        }
        return state;
    }

    std::vector<int> legal_moves(const std::vector<int8_t>& board) const {
        std::vector<int> moves;
        for (int i = 0; i < board_size_; ++i) {
            if (board[i] == 0) {
                moves.push_back(i);
            }
        }
        return moves;
    }

    std::vector<int> candidate_moves(const std::vector<int8_t>& board) const {
        if (candidate_distance_ < 0) {
            return legal_moves(board);
        }
        std::vector<int> occupied;
        for (int i = 0; i < board_size_; ++i) {
            if (board[i] != 0) {
                occupied.push_back(i);
            }
        }
        if (occupied.empty()) {
            return {height_ / 2 * width_ + width_ / 2};
        }
        std::vector<uint8_t> mask(static_cast<size_t>(board_size_), 0);
        for (int move : occupied) {
            int row = move / width_;
            int col = move % width_;
            for (int r = std::max(0, row - candidate_distance_); r <= std::min(height_ - 1, row + candidate_distance_); ++r) {
                for (int c = std::max(0, col - candidate_distance_); c <= std::min(width_ - 1, col + candidate_distance_); ++c) {
                    int idx = r * width_ + c;
                    if (board[idx] == 0) {
                        mask[idx] = 1;
                    }
                }
            }
        }
        std::vector<int> moves;
        for (int i = 0; i < board_size_; ++i) {
            if (mask[i]) {
                moves.push_back(i);
            }
        }
        return moves.empty() ? legal_moves(board) : moves;
    }

    int find_winning_move(const std::vector<int8_t>& board, int player, const std::vector<int>& legal) const {
        std::vector<int8_t> probe = board;
        for (int move : legal) {
            probe[move] = static_cast<int8_t>(player);
            bool wins = has_five_from(probe, move, player);
            probe[move] = 0;
            if (wins) {
                return move;
            }
        }
        return -1;
    }

    bool has_five_from(const std::vector<int8_t>& board, int move, int player) const {
        int row = move / width_;
        int col = move % width_;
        if (move < 0 || move >= board_size_ || board[move] != player) {
            return false;
        }
        constexpr int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        for (const auto& dir : dirs) {
            int count = 1;
            count += count_direction(board, row, col, dir[0], dir[1], player);
            count += count_direction(board, row, col, -dir[0], -dir[1], player);
            if (count >= 5) {
                return true;
            }
        }
        return false;
    }

    int count_direction(const std::vector<int8_t>& board, int row, int col, int dr, int dc, int player) const {
        int count = 0;
        int r = row + dr;
        int c = col + dc;
        while (r >= 0 && r < height_ && c >= 0 && c < width_ && board[r * width_ + c] == player) {
            ++count;
            r += dr;
            c += dc;
        }
        return count;
    }

    int opponent(int player) const {
        return player == first_player_ ? second_player_ : first_player_;
    }

    int height_;
    int width_;
    int board_size_;
    int first_player_;
    int second_player_;
    int n_playout_;
    double c_puct_;
    int candidate_distance_;
    bool tactical_shortcuts_;
    int current_player_ = 0;
    int root_ = 0;
    int next_request_id_ = 1;
    std::vector<int8_t> board_;
    std::vector<Node> nodes_;
    std::unordered_map<int, PendingRequest> pending_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<CppMCTSGame>(m, "CppMCTSGame")
        .def(py::init<int, int, int, int, int, double, int, bool>())
        .def("reset", &CppMCTSGame::reset)
        .def("request_leaf", &CppMCTSGame::request_leaf)
        .def("apply_evaluation", &CppMCTSGame::apply_evaluation)
        .def("action_probs", &CppMCTSGame::action_probs)
        .def("tactical_move", &CppMCTSGame::tactical_move)
        .def("apply_move", &CppMCTSGame::apply_move)
        .def("game_end_after_move", &CppMCTSGame::game_end_after_move)
        .def("board", &CppMCTSGame::board)
        .def("current_player", &CppMCTSGame::current_player)
        .def("root_visits", &CppMCTSGame::root_visits)
        .def("n_playout", &CppMCTSGame::n_playout);
}
