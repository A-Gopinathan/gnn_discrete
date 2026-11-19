#include "potts_model.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
static volatile std::sig_atomic_t interrupted = 0;
static void on_signal(int signal) { interrupted = signal; }
static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
static void syscheck(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message + ": " + std::strerror(errno));
}
struct FD {
    int value;
    explicit FD(int fd) : value(fd) { syscheck(fd >= 0, "open"); }
    ~FD() { if (value >= 0) ::close(value); }
    FD(const FD&) = delete;
    FD& operator=(const FD&) = delete;
};
static void sync_directory(const fs::path& path) {
    FD fd(::open(path.c_str(), O_RDONLY | O_DIRECTORY));
    syscheck(::fsync(fd.value) == 0, "fsync directory " + path.string());
}
static void write_all(int fd, const char* data, size_t size) {
    while (size) {
        ssize_t n = ::write(fd, data, size);
        if (n < 0 && errno == EINTR) continue;
        syscheck(n > 0, "write");
        data += n;
        size -= static_cast<size_t>(n);
    }
}
static std::string read_file(const fs::path& path) {
    require(fs::is_regular_file(fs::symlink_status(path)), "not a regular file: " + path.string());
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot read " + path.string());
    std::ostringstream out;
    out << in.rdbuf();
    require(!in.bad(), "read failed: " + path.string());
    return out.str();
}
static void atomic_text(const fs::path& path, const std::string& data) {
    fs::path tmp = path.string() + ".tmp";
    FD fd(::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600));
    write_all(fd.value, data.data(), data.size());
    syscheck(::fsync(fd.value) == 0, "fsync " + tmp.string());
    syscheck(::rename(tmp.c_str(), path.c_str()) == 0, "rename " + path.string());
    sync_directory(path.parent_path());
}
static uint64_t hash_update(uint64_t h, const std::string& data) {
    for (unsigned char c : data) { h ^= c; h *= UINT64_C(1099511628211); }
    return h;
}
static std::string hash_bytes(const std::string& data) {
    const uint64_t h = hash_update(UINT64_C(14695981039346656037), data);
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << h;
    return out.str();
}
static uint64_t integer(const std::string& s) {
    require(!s.empty() && s.find_first_not_of("0123456789") == std::string::npos,
            "invalid nonnegative integer: " + s);
    size_t used = 0;
    uint64_t n = std::stoull(s, &used);
    require(used == s.size(), "invalid integer");
    return n;
}
static double real(const std::string& s) {
    size_t used = 0;
    double n = std::stod(s, &used);
    require(used == s.size() && std::isfinite(n), "invalid finite number: " + s);
    return n;
}
struct Options {
    fs::path campaign, output;
    int trial = -1;
    bool local = false;
    double max_seconds = 0, checkpoint_seconds = 60;
    uint64_t stop_attempts = 0;
};
static Options options(int argc, char** argv) {
    Options o;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        require(seen.insert(k).second, "duplicate option: " + k);
        if (k == "--allow-local-test") { o.local = true; continue; }
        require(i + 1 < argc, "missing value for " + k);
        std::string v = argv[++i];
        if (k == "--campaign") o.campaign = v;
        else if (k == "--output") o.output = v;
        else if (k == "--trial") {
            auto n = integer(v);
            require(n < 100, "trial must be in [0,chains-1], chains<=100");
            o.trial = static_cast<int>(n);
        } else if (k == "--max-seconds") o.max_seconds = real(v);
        else if (k == "--checkpoint-seconds") o.checkpoint_seconds = real(v);
        else if (k == "--stop-after-attempts") o.stop_attempts = integer(v);
        else throw std::runtime_error("unknown option: " + k);
    }
    require(!o.campaign.empty() && !o.output.empty() && o.trial >= 0,
            "usage: potts_dynamics_ed --campaign DIR --output ROOT --trial ID "
            "[--max-seconds FLOAT] [--stop-after-attempts INT] [--checkpoint-seconds FLOAT] [--allow-local-test]");
    require(o.max_seconds >= 0 && o.checkpoint_seconds > 0, "invalid time limit/interval");
    const char* job = std::getenv("SLURM_JOB_ID");
    require((job && *job) || o.local, "ED requires SLURM_JOB_ID (including tests)");
    return o;
}
struct Config {
    int length, sweeps, chains;
    std::string campaign_id, fingerprint;
};
static Config config(const Options& o) {
    std::map<std::string, std::string> m;
    std::istringstream in(read_file(o.campaign / "campaign.cfg"));
    std::string line;
    while (std::getline(in, line)) {
        require(!line.empty() && line.find_first_of(" \t\r") == std::string::npos,
                "campaign.cfg requires nonempty key=value lines without whitespace");
        size_t pos = line.find('=');
        require(pos != std::string::npos && pos > 0 && pos + 1 < line.size() &&
                line.find('=', pos + 1) == std::string::npos, "invalid config line");
        require(m.emplace(line.substr(0, pos), line.substr(pos + 1)).second, "duplicate config key");
    }
    const std::map<std::string, std::string> fixed = {
        {"format", "POTTS_DYNAMICS_V1"},
        {"temperature", "0.1"}, {"filling", "0.5"}, {"alpha", "1"},
        {"beta", "-0.5"}, {"gamma", "-0.7"},
        {"initial_seed_base", "2026101100"}, {"ed_seed_base", "2026201100"},
        {"ml_seed_base", "2026301100"}};
    require(m.size() == fixed.size() + 4, "missing or unknown config keys");
    for (const auto& entry : fixed) {
        require(m.count(entry.first), "missing config value: " + entry.first);
        const bool physics = entry.first == "temperature" || entry.first == "filling" ||
                             entry.first == "alpha" || entry.first == "beta" || entry.first == "gamma";
        require(physics ? real(m.at(entry.first)) == real(entry.second) : m.at(entry.first) == entry.second,
                "unsupported config value: " + entry.first);
    }
    for (const auto& key : {"length", "sweeps", "chains", "campaign_id"})
        require(m.count(key), std::string("missing config key: ") + key);
    uint64_t L = integer(m.at("length")), sweeps = integer(m.at("sweeps"));
    uint64_t chains = integer(m.at("chains"));
    require(L >= 4 && L <= 128 && L % 2 == 0, "length must be even and in [4,128]");
    require(sweeps <= 100000 && chains >= 1 && chains <= 100 && uint64_t(o.trial) < chains,
            "invalid sweeps/chains or trial outside [0,chains-1]");
    require(!o.local || (L <= 4 && sweeps <= 2), "--allow-local-test requires L4 and <=2 sweeps");
    const auto& id = m.at("campaign_id");
    require(id.size() == 64 && id.find_first_not_of("0123456789abcdef") == std::string::npos,
            "campaign_id must be lowercase SHA256 hex");
    std::string canonical;
    for (const auto& entry : m) canonical += entry.first + "=" + entry.second + "\n";
    return {static_cast<int>(L), static_cast<int>(sweeps), static_cast<int>(chains), id, hash_bytes(canonical)};
}
static void validate_states(const std::string& s, size_t n) {
    require(s.size() == n, "wrong lattice byte count");
    for (unsigned char c : s) require(c <= 2, "state is not 0, 1 or 2");
}
struct Row { uint64_t changed; double elapsed, energy; };
struct State {
    int sweep = 0, within = 0;
    uint64_t changed = 0;
    uint64_t raw_hash = UINT64_C(14695981039346656037);
    double elapsed = 0, energy = 0;
    std::string current, completed;
    std::mt19937 rng;
    std::vector<Row> history;
};
static void expect(std::istream& in, const std::string& word) {
    std::string got;
    require(bool(in >> got) && got == word, "checkpoint expected " + word);
}
static std::string checkpoint(const State& s, const Config& c, int trial, const std::string& initial_hash) {
    std::ostringstream out;
    out << std::setprecision(17)
        << "identity " << c.fingerprint << ' ' << c.campaign_id << ' ' << initial_hash << ' ' << trial << '\n'
        << "position " << s.sweep << ' ' << s.within << ' ' << s.changed << ' ' << s.elapsed << ' ' << s.energy << '\n'
        << "raw_hash " << s.raw_hash << '\n'
        << "current";
    for (unsigned char x : s.current) out << ' ' << unsigned(x);
    out << "\ncompleted";
    for (unsigned char x : s.completed) out << ' ' << unsigned(x);
    out << "\nrng " << s.rng << "\nhistory " << s.history.size() << '\n';
    for (size_t i = 0; i < s.history.size(); ++i)
        out << i << ' ' << s.history[i].changed << ' ' << s.history[i].elapsed << ' ' << s.history[i].energy << '\n';
    out << "END\n";
    std::string body = out.str();
    return "POTTS_DYNAMICS_ED_CHECKPOINT_V1\n" + hash_bytes(body) + "\n" + body;
}
static State restore(const fs::path& path, const Config& c, int trial, const std::string& initial_hash, int n) {
    std::string text = read_file(path);
    size_t first = text.find('\n'), second = text.find('\n', first == std::string::npos ? 0 : first + 1);
    require(first != std::string::npos && second != std::string::npos &&
            text.substr(0, first) == "POTTS_DYNAMICS_ED_CHECKPOINT_V1", "bad checkpoint header");
    std::string body = text.substr(second + 1);
    require(text.substr(first + 1, second - first - 1) == hash_bytes(body), "checkpoint checksum mismatch");
    std::istringstream in(body);
    expect(in, "identity");
    std::string fingerprint, id, hash;
    int saved_trial = -1;
    in >> fingerprint >> id >> hash >> saved_trial;
    require(in && fingerprint == c.fingerprint && id == c.campaign_id && hash == initial_hash && saved_trial == trial,
            "checkpoint campaign/config/initial/trial mismatch");
    State s;
    expect(in, "position");
    in >> s.sweep >> s.within >> s.changed >> s.elapsed >> s.energy;
    require(in && s.sweep >= 0 && s.sweep <= c.sweeps && s.within >= 0 && s.within < n &&
            (s.sweep < c.sweeps || s.within == 0) && std::isfinite(s.elapsed) && s.elapsed >= 0 &&
             std::isfinite(s.energy) && s.changed <= uint64_t(s.sweep) * n + s.within, "invalid checkpoint position");
    expect(in, "raw_hash");
    require(bool(in >> s.raw_hash), "invalid raw trajectory hash");
    const std::array<std::pair<const char*, std::string*>, 2> lattices = {{{"current", &s.current}, {"completed", &s.completed}}};
    for (const auto& entry : lattices) {
        expect(in, entry.first);
        for (int i = 0; i < n; ++i) {
            int x = -1;
            require(bool(in >> x) && x >= 0 && x <= 2, "invalid checkpoint lattice");
            entry.second->push_back(static_cast<char>(x));
        }
    }
    require(s.within != 0 || s.current == s.completed, "boundary lattice mismatch");
    expect(in, "rng");
    require(bool(in >> s.rng), "invalid checkpoint RNG");
    expect(in, "history");
    size_t count = 0;
    require(bool(in >> count) && count == size_t(s.sweep + 1), "invalid history length");
    for (size_t i = 0; i < count; ++i) {
        size_t index = count;
        Row r{};
        require(bool(in >> index >> r.changed >> r.elapsed >> r.energy) && index == i &&
                std::isfinite(r.elapsed) && r.elapsed >= 0 && r.elapsed <= s.elapsed &&
                std::isfinite(r.energy) && r.changed <= i * n && r.changed <= s.changed, "invalid history row");
        if (i) require(r.changed >= s.history.back().changed && r.changed - s.history.back().changed <= uint64_t(n) &&
                       r.elapsed >= s.history.back().elapsed, "nonmonotonic history");
        s.history.push_back(r);
    }
    require(s.changed - s.history.back().changed <= uint64_t(s.within), "invalid current change count");
    require(s.within != 0 || s.energy == s.history.back().energy, "boundary energy mismatch");
    expect(in, "END");
    in >> std::ws;
    require(in.eof(), "trailing checkpoint data");
    return s;
}
static std::string metadata(const Config& c, int trial, const std::string& hash) {
    std::ostringstream out;
    out << "{\n  \"format\": \"POTTS_DYNAMICS_V1\",\n  \"track\": \"ed\",\n"
        << "  \"campaign_id\": \"" << c.campaign_id << "\",\n"
        << "  \"config_fingerprint_fnv1a64\": \"" << c.fingerprint << "\",\n"
        << "  \"initial_hash_fnv1a64\": \"" << hash << "\",\n"
        << "  \"trial\": " << trial << ",\n  \"linear_size\": " << c.length
        << ",\n  \"total_sweeps\": " << c.sweeps << ",\n  \"chains\": " << c.chains << ",\n"
        << "  \"initial_seed\": " << 2026101100ULL + trial << ",\n"
        << "  \"ed_seed\": " << 2026201100ULL + trial << ",\n"
        << "  \"ml_seed\": " << 2026301100ULL + trial << ",\n"
        << "  \"physics\": {\"temperature\": 0.1, \"filling\": 0.5, \"alpha\": 1, \"beta\": -0.5, \"gamma\": -0.7},\n"
        << "  \"frame_convention\": \"uint8[completed_sweeps+1,L,L,L], C-order x*L*L+y*L+z; frame 0 initial; states 0,1,2\",\n"
        << "  \"update\": \"random site with replacement; L^3 attempts/sweep; sample all 3 states with stable heat-bath weights\",\n"
        << "  \"energy\": \"total electronic internal energy; ED eigenvalues-only; cached current energy; two candidate ED per attempt; no classical energy\",\n"
        << "  \"rng\": \"std::mt19937; uniform_int_distribution site; generate_canonical<double,53> heat-bath draw\"\n}\n";
    return out.str();
}
static void csv(const fs::path& dir, const State& s, int n) {
    std::ostringstream progress, energies;
    progress << std::setprecision(17) << "sweep,attempts,changed_sites,elapsed_seconds\n";
    energies << std::setprecision(17) << "sweep,total_energy,energy_per_site\n";
    for (size_t i = 0; i < s.history.size(); ++i) {
        const Row& r = s.history[i];
        progress << i << ',' << i * n << ',' << r.changed << ',' << r.elapsed << '\n';
        energies << i << ',' << r.energy << ',' << r.energy / n << '\n';
    }
    atomic_text(dir / "progress.csv", progress.str());
    atomic_text(dir / "energy.csv", energies.str());
}
int main(int argc, char** argv) {
    try {
        const Options o = options(argc, argv);
        const Config c = config(o);
        std::signal(SIGUSR1, on_signal);
        std::signal(SIGTERM, on_signal);
        const auto start = Clock::now();
        auto seconds = [&]() { return std::chrono::duration<double>(Clock::now() - start).count(); };
        std::ostringstream name;
        name << "trial_" << std::setw(3) << std::setfill('0') << o.trial;
        const int n = c.length * c.length * c.length;
        const std::string initial = read_file(o.campaign / "initial_states" / (name.str() + ".bin"));
        validate_states(initial, n);
        const std::string initial_hash = hash_bytes(initial);
        fs::create_directories(o.output);
        FD lock(::open((o.output / (name.str() + ".lock")).c_str(), O_RDWR | O_CREAT | O_NOFOLLOW, 0600));
        syscheck(::flock(lock.value, LOCK_EX | LOCK_NB) == 0, "trial is locked");
        const fs::path dir = o.output / name.str();
        if (fs::exists(dir)) require(fs::is_directory(fs::symlink_status(dir)), "trial path is not a regular directory");
        else { fs::create_directory(dir); sync_directory(o.output); }
        const std::set<std::string> allowed = {"metadata.json", "checkpoint.dat", "states.bin", "progress.csv", "energy.csv", "COMPLETE",
            "metadata.json.tmp", "checkpoint.dat.tmp", "progress.csv.tmp", "energy.csv.tmp", "COMPLETE.tmp"};
        for (const auto& entry : fs::directory_iterator(dir)) {
            require(allowed.count(entry.path().filename().string()) && fs::is_regular_file(entry.symlink_status()),
                    "unexpected file in trial directory: " + entry.path().string());
        }
        const std::string meta = metadata(c, o.trial, initial_hash);
        if (fs::exists(dir / "metadata.json"))
            require(read_file(dir / "metadata.json") == meta, "metadata mismatch");
        const bool resume = fs::exists(dir / "checkpoint.dat");
        State s;
        if (resume) {
            s = restore(dir / "checkpoint.dat", c, o.trial, initial_hash, n);
            std::string raw = read_file(dir / "states.bin");
            const size_t committed = size_t(s.sweep + 1) * n;
            require(raw.size() >= committed, "states.bin shorter than committed checkpoint frames");
            require(raw.compare(0, n, initial) == 0, "raw initial frame mismatch");
            validate_states(raw.substr(0, committed), committed);
            require(hash_update(UINT64_C(14695981039346656037), raw.substr(0, committed)) == s.raw_hash,
                    "committed raw trajectory checksum mismatch");
            require(raw.compare(size_t(s.sweep) * n, n, s.completed) == 0, "last committed raw frame/checkpoint mismatch");
        } else {
            require(!fs::exists(dir / "COMPLETE") && !fs::exists(dir / "progress.csv") && !fs::exists(dir / "energy.csv"),
                    "outputs exist without checkpoint; refusing overwrite");
            if (fs::exists(dir / "states.bin"))
                require(read_file(dir / "states.bin") == initial, "no checkpoint and states.bin is not pristine frame 0; manual recovery needed");
            s.current = s.completed = initial;
            s.raw_hash = hash_update(s.raw_hash, initial);
            s.rng.seed(static_cast<uint32_t>(2026201100ULL + o.trial));
        }
        const std::string marker = "POTTS_DYNAMICS_ED_COMPLETE_V1\n" + c.campaign_id + "\n" +
            c.fingerprint + " " + initial_hash + " " + std::to_string(o.trial) + " " + std::to_string(c.sweeps) + "\n";
        if (fs::exists(dir / "COMPLETE"))
            require(resume && s.sweep == c.sweeps && s.within == 0 && read_file(dir / "COMPLETE") == marker,
                    "COMPLETE disagrees with checkpoint");
        Cubic_Lattice lattice(c.length);
        lattice.kT = 0.1;
        lattice.filling = 0.5;
        for (int i = 0; i < n; ++i) lattice.site[i].state = static_cast<unsigned char>(s.current[i]);
        const double energy = lattice.compute_energy_silent();
        require(std::isfinite(energy), "initial/resume ED returned nonfinite energy");
        if (resume) require(std::abs(energy - s.energy) <= 1e-8 + 1e-8 * std::abs(s.energy), "resume ED energy mismatch");
        else s.energy = energy;
        const double previous_elapsed = s.elapsed;
        auto elapsed = [&]() { return previous_elapsed + seconds(); };
        if (!resume) s.history.push_back({0, elapsed(), s.energy});
        FD raw(::open((dir / "states.bin").c_str(), O_RDWR | O_CREAT | O_NOFOLLOW, 0600));
        const off_t committed = static_cast<off_t>(s.sweep + 1) * n;
        if (resume) syscheck(::ftruncate(raw.value, committed) == 0, "truncate uncommitted raw bytes");
        else { syscheck(::lseek(raw.value, 0, SEEK_SET) == 0, "seek initial frame"); write_all(raw.value, initial.data(), initial.size()); }
        syscheck(::lseek(raw.value, committed, SEEK_SET) == committed, "seek committed raw end");
        syscheck(::fsync(raw.value) == 0, "fsync raw frames");
        if (!fs::exists(dir / "metadata.json")) atomic_text(dir / "metadata.json", meta);
        double last_checkpoint = seconds();
        uint64_t invocation_attempts = 0;
        auto save = [&]() {
            // The raw frame must be durable before publishing its checkpoint commitment.
            syscheck(::fsync(raw.value) == 0, "fsync raw frames");
            s.elapsed = elapsed();
            atomic_text(dir / "checkpoint.dat", checkpoint(s, c, o.trial, initial_hash));
            csv(dir, s, n);
            last_checkpoint = seconds();
            const uint64_t attempts = uint64_t(s.sweep) * n + s.within;
            const double rate = invocation_attempts / std::max(seconds(), 1e-12);
            std::cout << std::setprecision(10) << "checkpoint trial=" << o.trial << " sweep=" << s.sweep
                      << " attempt_in_sweep=" << s.within << " attempts=" << attempts << " changed_sites=" << s.changed
                      << " elapsed_seconds=" << s.elapsed << " attempts_per_second=" << rate
                      << " estimated_remaining_seconds=" << (rate > 0 ? (uint64_t(c.sweeps) * n - attempts) / rate : -1)
                      << std::endl;
        };
        save();
        std::uniform_int_distribution<int> site_distribution(0, n - 1);
        auto stopping = [&]() { return interrupted || (o.max_seconds > 0 && seconds() >= o.max_seconds) ||
                                      (o.stop_attempts && invocation_attempts >= o.stop_attempts); };
        while (s.sweep < c.sweeps && !stopping()) {
            const int site = site_distribution(s.rng);
            const int old = lattice.site[site].state;
            std::array<double, 3> energies{};
            energies[old] = s.energy;
            for (int state = 0; state < 3; ++state) if (state != old) {
                lattice.site[site].state = state;
                energies[state] = lattice.compute_energy_silent();
                require(std::isfinite(energies[state]), "candidate ED returned nonfinite energy");
            }
            const double minimum = *std::min_element(energies.begin(), energies.end());
            std::array<double, 3> weights{};
            double total = 0;
            for (int state = 0; state < 3; ++state) total += weights[state] = std::exp(-(energies[state] - minimum) / 0.1);
            const double draw = std::generate_canonical<double, 53>(s.rng) * total;
            int selected = 2;
            if (draw < weights[0]) selected = 0;
            else if (draw < weights[0] + weights[1]) selected = 1;
            lattice.site[site].state = selected;
            s.current[site] = static_cast<char>(selected);
            s.energy = energies[selected];
            s.changed += selected != old;
            ++s.within;
            ++invocation_attempts;
            const bool boundary = s.within == n;
            if (boundary) {
                ++s.sweep;
                s.within = 0;
                s.completed = s.current;
                write_all(raw.value, s.completed.data(), s.completed.size());
                s.raw_hash = hash_update(s.raw_hash, s.completed);
                s.history.push_back({s.changed, elapsed(), s.energy});
            }
            if (boundary || seconds() - last_checkpoint >= o.checkpoint_seconds || stopping()) save();
        }
        save();
        if (s.sweep == c.sweeps && s.within == 0) {
            atomic_text(dir / "COMPLETE", marker);
            std::cout << "status=complete trial=" << o.trial << std::endl;
        } else std::cout << "status=stopped trial=" << o.trial << " signal=" << interrupted << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}
