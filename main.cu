#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

// Portable counter-based PRNG
#include <Random123/philox.h>
#include <Random123/boxmuller.hpp>

#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <stdexcept>

// ============================================================================
// CONFIGURATION: Set to 0 for O(N^2) Direct Force or 1 for O(N) Cell List
// ============================================================================
#ifndef USE_CELL_LIST
#define USE_CELL_LIST 1
#endif

#define s3 1.7320508075688772f

// Soft-core radius: the K1 pair force diverges as 1/r, so separations below
// this are clamped. Physically this is the vortex core size.
#define R_CORE 1.0e-3f

// ============================================================================
// Special functions
// ============================================================================

// Modified Bessel I1, Abramowitz & Stegun 9.8.3 / 9.8.4.
__host__ __device__ inline float bessel_i1(float x) {
    float ax = fabsf(x);
    if (ax < 3.75f) {
        float t = x / 3.75f;
        float t2 = t * t;
        float p = 0.5f + t2 * (0.87890594f + t2 * (0.51498869f + t2 * (0.15084934f
                + t2 * (0.02658733f + t2 * (0.00301532f + t2 * 0.00032411f)))));
        return x * p;
    }
    float t = 3.75f / ax;
    float p = 0.02282967f + t * (-0.02895312f + t * (0.01787654f - t * 0.00420059f));
    p = 0.39894228f + t * (-0.03988024f + t * (-0.00362018f + t * (0.00163801f
      + t * (-0.01031555f + p))));
    float ans = p * expf(ax) / sqrtf(ax);
    return (x < 0.0f) ? -ans : ans;
}

// Modified Bessel K1, Abramowitz & Stegun 9.8.7 / 9.8.8.
// Accurate to ~1e-7 relative and continuous across the x = 2 branch point.
__host__ __device__ inline float k1_approx(float x) {
    if (x < R_CORE) x = R_CORE;           // core cutoff, avoids the singularity
    if (x <= 2.0f) {
        float y = x * x * 0.25f;          // (x/2)^2
        float p = 1.0f + y * (0.15443144f + y * (-0.67278579f + y * (-0.18156897f
                + y * (-0.01919402f + y * (-0.00110404f - y * 0.00004686f)))));
        return logf(0.5f * x) * bessel_i1(x) + p / x;
    }
    float y = 2.0f / x;
    float p = 1.25331414f + y * (0.23498619f + y * (-0.03655620f + y * (0.01504268f
            + y * (-0.00780353f + y * (0.00325614f - y * 0.00068245f)))));
    return p * expf(-x) / sqrtf(x);
}

// ============================================================================
// Periodic boundary helpers
// ============================================================================

// Minimum image convention (valid for cutoff <= L/2)
__host__ __device__ inline float min_dist(float d, float L) {
    return d - L * roundf(d / L);
}

// Fold a coordinate into [0, L)
__host__ __device__ inline float fold(float x, float L) {
    float r = x - L * floorf(x / L);
    if (r >= L) r = 0.0f;                 // guard against float round-up at the edge
    if (r < 0.0f) r = 0.0f;
    return r;
}

// ============================================================================
// Initialisation
// ============================================================================

// Initial triangular lattice. The identity stored is the GLOBAL particle id
// (= idx = z * Nxy + in-plane label), which is unique across the whole system.
// The in-plane label is recovered as gid % Nxy, the layer as gid / Nxy.
struct InitTriangularLattice {
    int Nx, Nxy;
    float a0, Lx, Ly;

    __host__ __device__
    thrust::tuple<float, float, int, int> operator()(int idx) const {
        int z = idx / Nxy;
        int vortex_id = idx % Nxy;
        int row = vortex_id / Nx;
        int col = vortex_id % Nx;

        float y = row * a0 * 0.5f * s3;
        float x = col * a0 + (row % 2) * a0 * 0.5f;

        return thrust::make_tuple(fold(x, Lx), fold(y, Ly), z, idx);
    }
};

// Compute cell ID for domain decomposition.
struct GetCellID {
    int Ncx, Ncy, Ncxy;
    float cut_offx, cut_offy, Lx, Ly;

    __host__ __device__
    int operator()(const thrust::tuple<float, float, int>& t) const {
        float x = fold(thrust::get<0>(t), Lx);
        float y = fold(thrust::get<1>(t), Ly);
        int z = thrust::get<2>(t);

        int cx = int(x / cut_offx);
        int cy = int(y / cut_offy);
        // Clamp: a coordinate rounding up to L must not produce an out-of-range cell
        if (cx >= Ncx) cx = Ncx - 1;
        if (cy >= Ncy) cy = Ncy - 1;
        if (cx < 0) cx = 0;
        if (cy < 0) cy = 0;

        return cx + cy * Ncx + z * Ncxy;
    }
};

// ============================================================================
// Verlet-skin cell list
//
// The mesh cells are sized to cutoff+skin (see GetCellID call sites), so a
// pair currently within `cutoff` was, at the last rebuild, within cutoff+skin
// (separation changes by at most `skin` while each particle drifts at most
// skin/2). Rebuilding whenever any particle's drift since the last build
// exceeds skin/2 therefore guarantees the stale cell assignment still finds
// every true neighbor, without re-sorting every step.
// ============================================================================

// Squared displacement of a particle since the last cell-list rebuild.
struct DisplacementSq {
    float Lx, Ly;

    __host__ __device__
    float operator()(const thrust::tuple<float, float, float, float>& t) const {
        float dx = min_dist(thrust::get<0>(t) - thrust::get<2>(t), Lx);
        float dy = min_dist(thrust::get<1>(t) - thrust::get<3>(t), Ly);
        return dx * dx + dy * dy;
    }
};

// Re-sort particles into cells and rebuild the cell offset table.
void rebuildCellList(
    thrust::device_vector<float>& posx,
    thrust::device_vector<float>& posy,
    thrust::device_vector<int>& posz,
    thrust::device_vector<int>& which_vortex_is,
    thrust::device_vector<int>& where_vortex_is,
    thrust::device_vector<int>& cell_ids,
    thrust::device_vector<int>& cell_starts,
    thrust::device_vector<int>& cell_ends,
    const thrust::device_vector<int>& cell_range,
    int Ncx, int Ncy, int Ncxy,
    float cut_offx, float cut_offy, float Lx, float Ly,
    int Nmax)
{
    // Assign cell IDs
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(posx.begin(), posy.begin(), posz.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(posx.end(), posy.end(), posz.end())),
        cell_ids.begin(),
        GetCellID{Ncx, Ncy, Ncxy, cut_offx, cut_offy, Lx, Ly}
    );

    // Sort particle data by cell ID for spatial locality
    auto particle_data = thrust::make_zip_iterator(
        thrust::make_tuple(posx.begin(), posy.begin(), posz.begin(), which_vortex_is.begin())
    );
    thrust::sort_by_key(cell_ids.begin(), cell_ids.end(), particle_data);

    // Rebuild the inverse map: where_vortex_is[which_vortex_is[i]] = i.
    // Global ids are unique, so a scatter is exact and cheaper than a sort.
    thrust::scatter(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(Nmax),
        which_vortex_is.begin(),
        where_vortex_is.begin()
    );

    // Construct contiguous cell offsets
    thrust::lower_bound(cell_ids.begin(), cell_ids.end(),
                        cell_range.begin(), cell_range.end(), cell_starts.begin());
    thrust::upper_bound(cell_ids.begin(), cell_ids.end(),
                        cell_range.begin(), cell_range.end(), cell_ends.begin());
}

// ============================================================================
// Force evaluation
// ============================================================================

// Inter-layer harmonic springs, periodic in z.
// which_vortex_is[slot] -> global id;  where_vortex_is[global id] -> slot.
__host__ __device__ inline void add_spring_forces(
    const float* posx, const float* posy,
    const int* where_vortex_is,
    int idx, int gid, int Nxy, int Nz,
    float Lx, float Ly, float k_spring,
    float& fx, float& fy)
{
    int label = gid % Nxy;                // in-plane identity of the vortex line
    int z1    = gid / Nxy;                // current layer

    for (int dz = -1; dz <= 1; dz += 2) {
        int nplane  = (z1 + dz + Nz) % Nz;               // PBC along z
        int partner = where_vortex_is[label + nplane * Nxy];

        fx += k_spring * min_dist(posx[partner] - posx[idx], Lx);
        fy += k_spring * min_dist(posy[partner] - posy[idx], Ly);
    }
}

// METHOD 1: Direct O(N^2) force calculation
struct CalculateForcesDirect {
    const float* posx;
    const float* posy;
    const int* posz;
    const int* which_vortex_is;
    const int* where_vortex_is;
    int Nmax, Nxy, Nz;
    float Lx, Ly, k_spring, cutoff2;

    __host__ __device__
    thrust::tuple<float, float> operator()(int idx) const {
        float x1 = posx[idx];
        float y1 = posy[idx];
        int z1   = posz[idx];
        int gid  = which_vortex_is[idx];

        float fx = 0.0f, fy = 0.0f;

        // In-plane pair interactions (all particles in the same layer)
        int plane_start = z1 * Nxy;
        for (int i = 0; i < Nxy; ++i) {
            int nidx = plane_start + i;
            if (nidx == idx) continue;

            float rx = min_dist(x1 - posx[nidx], Lx);
            float ry = min_dist(y1 - posy[nidx], Ly);
            float r2 = rx * rx + ry * ry;

            if (r2 < cutoff2) {
                float r = sqrtf(r2);
                if (r < R_CORE) r = R_CORE;
                float f = k1_approx(r) / r;
                fx += f * rx;
                fy += f * ry;
            }
        }

        add_spring_forces(posx, posy, where_vortex_is,
                          idx, gid, Nxy, Nz, Lx, Ly, k_spring, fx, fy);

        return thrust::make_tuple(fx, fy);
    }
};

// METHOD 2: O(N) cell list
struct CalculateForcesCellList {
    const float* posx;
    const float* posy;
    const int* posz;
    const int* cell_starts;
    const int* cell_ends;
    const int* which_vortex_is;
    const int* where_vortex_is;
    int Nmax, Nxy, Nz, Ncx, Ncy, Ncxy;
    float Lx, Ly, k_spring, cutoff2, cut_offx, cut_offy;

    __host__ __device__
    thrust::tuple<float, float> operator()(int idx) const {
        float x1 = posx[idx];
        float y1 = posy[idx];
        int z1   = posz[idx];
        int gid  = which_vortex_is[idx];

        float x1_box = fold(x1, Lx);
        float y1_box = fold(y1, Ly);

        int cx = int(x1_box / cut_offx);
        int cy = int(y1_box / cut_offy);
        if (cx >= Ncx) cx = Ncx - 1;
        if (cy >= Ncy) cy = Ncy - 1;
        if (cx < 0) cx = 0;
        if (cy < 0) cy = 0;

        float fx = 0.0f, fy = 0.0f;

        // Sweep the 3x3 in-plane neighbourhood. Cells are de-duplicated so that
        // a mesh with only one or two cells along an axis is not visited twice.
        int visited[9];
        int nvisited = 0;

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                int ncx = ((cx + dx) % Ncx + Ncx) % Ncx;   // PBC on the cell grid
                int ncy = ((cy + dy) % Ncy + Ncy) % Ncy;
                int ncell = ncx + ncy * Ncx + z1 * Ncxy;

                bool dup = false;
                for (int q = 0; q < nvisited; ++q)
                    if (visited[q] == ncell) { dup = true; break; }
                if (dup) continue;
                visited[nvisited++] = ncell;

                int start = cell_starts[ncell];
                int end   = cell_ends[ncell];

                for (int j = start; j < end; ++j) {
                    if (j == idx) continue;

                    float rx = min_dist(x1 - posx[j], Lx);
                    float ry = min_dist(y1 - posy[j], Ly);
                    float r2 = rx * rx + ry * ry;

                    if (r2 < cutoff2) {
                        float r = sqrtf(r2);
                        if (r < R_CORE) r = R_CORE;
                        float f = k1_approx(r) / r;
                        fx += f * rx;
                        fy += f * ry;
                    }
                }
            }
        }

        add_spring_forces(posx, posy, where_vortex_is,
                          idx, gid, Nxy, Nz, Lx, Ly, k_spring, fx, fy);

        return thrust::make_tuple(fx, fy);
    }
};

// ============================================================================
// Integrator
// ============================================================================

// Overdamped Langevin (Brownian) step, with the updated position folded back
// into the periodic box.
struct LangevinStep {
    const int* which_vortex_is;
    float dt, s2Tdt, Lx, Ly;
    unsigned long long seed, step;

    __host__ __device__
    thrust::tuple<float, float> operator()(
        const thrust::tuple<float, float, float, float, int>& t) const
    {
        float x  = thrust::get<0>(t);
        float y  = thrust::get<1>(t);
        float fx = thrust::get<2>(t);
        float fy = thrust::get<3>(t);
        int idx  = thrust::get<4>(t);

        // Persistent, globally unique vortex identity regardless of memory sorting,
        // so each particle draws its own independent noise stream.
        int gid = which_vortex_is[idx];

        r123::Philox4x32::key_type key = {{
            static_cast<uint32_t>(gid),
            0x9E3779B9u
        }};
        r123::Philox4x32::ctr_type ctr = {{
            static_cast<uint32_t>(step & 0xFFFFFFFFull),
            static_cast<uint32_t>(step >> 32),
            static_cast<uint32_t>(seed & 0xFFFFFFFFull),
            static_cast<uint32_t>(seed >> 32)
        }};

        r123::Philox4x32 rng_gen;
        r123::Philox4x32::ctr_type rng = rng_gen(ctr, key);

        r123::float2 noise = r123::boxmuller(rng[0], rng[1]);

        x += fx * dt + noise.x * s2Tdt;
        y += fy * dt + noise.y * s2Tdt;

        // Periodic boundary conditions in x and y (z is periodic by construction)
        x = fold(x, Lx);
        y = fold(y, Ly);

        return thrust::make_tuple(x, y);
    }
};

// ============================================================================
// I/O
// ============================================================================

void printConfiguration(
    const std::string& filename,
    int iter,
    float T,
    int Nmax,
    int Nxy,
    const thrust::device_vector<float>& posx,
    const thrust::device_vector<float>& posy,
    const thrust::device_vector<int>& posz,
    const thrust::device_vector<float>& fx,
    const thrust::device_vector<float>& fy,
    const thrust::device_vector<int>& which_vortex_is,
    const thrust::device_vector<int>& cell_ids
) {
    thrust::host_vector<float> h_posx = posx;
    thrust::host_vector<float> h_posy = posy;
    thrust::host_vector<int>   h_posz = posz;
    thrust::host_vector<float> h_fx   = fx;
    thrust::host_vector<float> h_fy   = fy;
    thrust::host_vector<int>   h_which = which_vortex_is;
    thrust::host_vector<int>   h_cells = cell_ids;

    bool has_cells = !h_cells.empty();

    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    fout << "Iter = " << iter << "\n";
    fout << "Temperatura = " << T << "\n";
    fout << "x\ty\tz\tfx\tfy\tvortex\tlabel\tcell\n";
    fout << std::fixed << std::setprecision(6);

    for (int i = 0; i < Nmax; ++i) {
        int cell_val = has_cells ? h_cells[i] : -1;
        int gid = h_which[i];

        fout << h_posx[i] << "\t"
             << h_posy[i] << "\t"
             << h_posz[i] << "\t"
             << h_fx[i]   << "\t"
             << h_fy[i]   << "\t"
             << gid       << "\t"
             << (gid % Nxy) << "\t"
             << cell_val  << "\n";

        // Separate 2D planes/layers with a double newline for easy gnuplot plotting
        if ((i + 1) % Nxy == 0) {
            fout << "\n\n";
        }
    }

    fout.close();
}

void writeSimulationLog(
    const std::string& filename,
    int Nx, int Ny, int Nz, int Nmax,
    float a0, float Lx, float Ly,
    float k_spring, float T, float dt, int steps, float cutoff,
    int Ncx, int Ncy, int Ncxy, int Nc, float cut_offx, float cut_offy,
    bool use_cell_list, float skin, float cutoff_skin
) {
    std::ofstream log(filename);
    if (!log.is_open()) {
        std::cerr << "Error: Could not create log file " << filename << "\n";
        return;
    }

    log << "====================================================\n";
    log << "       LANGEVIN VORTEX DYNAMICS SIMULATION LOG      \n";
    log << "====================================================\n\n";

    log << "[Execution Mode]\n";
    log << "Algorithm Mode       : " << (use_cell_list ? "Cell List O(N)" : "Direct O(N^2)") << "\n";
    log << "Boundary Conditions  : Periodic in x, y and z\n\n";

    log << "[System Geometry & Particles]\n";
    log << "Lattice Dimensions   : " << Nx << " x " << Ny << " x " << Nz << "\n";
    log << "In-Plane Vortices    : " << Nx * Ny << "\n";
    log << "Total Vortices (Nmax): " << Nmax << "\n";
    log << "Lattice Constant (a0): " << a0 << "\n";
    log << "Box Size Lx          : " << Lx << "\n";
    log << "Box Size Ly          : " << Ly << "\n\n";

    log << "[Physical Parameters]\n";
    log << "Temperature (T)      : " << T << "\n";
    log << "Time Step (dt)       : " << dt << "\n";
    log << "Total Steps          : " << steps << "\n";
    log << "Spring Constant (k)  : " << k_spring << "\n";
    log << "Interaction Cutoff   : " << cutoff << "  (absolute, independent of a0)\n";
    log << "Core Radius          : " << R_CORE << "\n\n";

    if (use_cell_list) {
        log << "[Cell List Mesh Parameters]\n";
        log << "Mesh Grid (Ncx, Ncy) : " << Ncx << " x " << Ncy << "\n";
        log << "Plane Cells (Ncxy)   : " << Ncxy << "\n";
        log << "Total 3D Cells (Nc)  : " << Nc << "\n";
        log << "Cell Width (dx)      : " << cut_offx << "\n";
        log << "Cell Height (dy)     : " << cut_offy << "\n\n";

        log << "[Verlet Skin List]\n";
        log << "Skin                 : " << skin << "\n";
        log << "Extended Cutoff      : " << cutoff_skin << "  (mesh is sized to this, not to the bare cutoff)\n";
        log << "Rebuild Criterion    : max single-particle drift since last rebuild > skin/2\n\n";
    }

    log << "====================================================\n";
    log.close();

    std::cout << "Simulation parameters logged to " << filename << std::endl;
}

// ============================================================================
// Command-line parameters
// ============================================================================

struct SimParams {
    int Nx = 20, Ny = 20, Nz = 4;
    float a0 = 1.0f;
    float k_spring = 0.5f;
    float T = 0.01f;
    float dt = 0.01f;
    int steps = 500;
    float cutoff = 3.0f;
    float skin = 0.5f;
    unsigned long long seed = 1234567ULL;
    int print_interval = 100;
};

void printUsage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options]\n"
        "  --nx N              in-plane lattice width          (default 20)\n"
        "  --ny N              in-plane lattice height         (default 20)\n"
        "  --nz N              number of layers                (default 4)\n"
        "  --a0 F              lattice constant                (default 1.0)\n"
        "  --k F               inter-layer spring constant     (default 0.5)\n"
        "  --T F               temperature                     (default 0.01)\n"
        "  --dt F              timestep                        (default 0.01)\n"
        "  --steps N           number of integration steps     (default 500)\n"
        "  --cutoff F          interaction cutoff radius       (default 3.0)\n"
        "  --skin F            Verlet skin width                (default 0.5)\n"
        "  --seed N            RNG seed                        (default 1234567)\n"
        "  --print-interval N  steps between snapshot writes   (default 100)\n"
        "  -h, --help          show this message\n";
}

// Returns 0 to proceed, 1 to exit successfully (help shown), -1 on bad input.
int parseArgs(int argc, char** argv, SimParams& p) {
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[++i];
            };

            if (arg == "-h" || arg == "--help") { printUsage(argv[0]); return 1; }
            else if (arg == "--nx") p.Nx = std::stoi(next());
            else if (arg == "--ny") p.Ny = std::stoi(next());
            else if (arg == "--nz") p.Nz = std::stoi(next());
            else if (arg == "--a0") p.a0 = std::stof(next());
            else if (arg == "--k") p.k_spring = std::stof(next());
            else if (arg == "--T") p.T = std::stof(next());
            else if (arg == "--dt") p.dt = std::stof(next());
            else if (arg == "--steps") p.steps = std::stoi(next());
            else if (arg == "--cutoff") p.cutoff = std::stof(next());
            else if (arg == "--skin") p.skin = std::stof(next());
            else if (arg == "--seed") p.seed = std::stoull(next());
            else if (arg == "--print-interval") p.print_interval = std::stoi(next());
            else throw std::invalid_argument("unknown option " + arg);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        printUsage(argv[0]);
        return -1;
    }
    return 0;
}

// ============================================================================

int main(int argc, char** argv) {
    SimParams p;
    int parse_rc = parseArgs(argc, argv, p);
    if (parse_rc != 0) return parse_rc > 0 ? EXIT_SUCCESS : EXIT_FAILURE;

    // --- Simulation Settings ---
    int Nx = p.Nx, Ny = p.Ny, Nz = p.Nz;
    int Nxy = Nx * Ny;
    int Nmax = Nxy * Nz;
    float a0 = p.a0;
    float k_spring = p.k_spring;
    float T = p.T;
    float dt = p.dt;
    int steps = p.steps;

    // Absolute interaction range, in the same length units as the box.
    // Deliberately NOT tied to a0: changing the lattice constant changes the
    // density, not the range of the interaction.
    float cutoff = p.cutoff;

    // Verlet skin: the cell-list mesh is built cutoff+skin wide so the list
    // stays a valid superset of the true neighbors for several steps. It is
    // rebuilt only once some particle has drifted more than skin/2 since the
    // last build (see the Verlet-skin cell list section above).
    float skin = p.skin;
    float cutoff_skin = cutoff + skin;
    float skin_half_sq = (skin * 0.5f) * (skin * 0.5f);

    float Lx = a0 * Nx;
    float Ly = a0 * Ny * 0.5f * s3;
    float s2Tdt = sqrtf(2.0f * T * dt);

    // The minimum image convention requires the extended (skin) cutoff to fit
    // inside half the box, since that is now the range the mesh geometry assumes.
    if (cutoff_skin > 0.5f * Lx || cutoff_skin > 0.5f * Ly) {
        std::cerr << "Error: cutoff+skin (" << cutoff_skin << ") exceeds half the box ("
                  << 0.5f * Lx << ", " << 0.5f * Ly << "). "
                  << "Increase Nx/Ny/a0 or reduce the cutoff/skin.\n";
        return EXIT_FAILURE;
    }

    // Cell list mesh: floor() guarantees every cell is at least `cutoff_skin`
    // wide, so the 3x3 sweep covers the full (skin-extended) interaction range.
    int Ncx = int(Lx / cutoff_skin); if (Ncx < 1) Ncx = 1;
    int Ncy = int(Ly / cutoff_skin); if (Ncy < 1) Ncy = 1;
    float cut_offx = Lx / Ncx;
    float cut_offy = Ly / Ncy;
    int Ncxy = Ncx * Ncy;
    int Nc = Ncxy * Nz;

    bool use_cell_list = (USE_CELL_LIST == 1);

    writeSimulationLog(
        "simulation.log",
        Nx, Ny, Nz, Nmax,
        a0, Lx, Ly,
        k_spring, T, dt, steps, cutoff,
        Ncx, Ncy, Ncxy, Nc, cut_offx, cut_offy,
        use_cell_list, skin, cutoff_skin
    );

    // --- Allocations ---
    // fx, fy hold FORCES (this integrator is overdamped; there are no velocities).
    thrust::device_vector<float> posx(Nmax), posy(Nmax), fx(Nmax), fy(Nmax);
    thrust::device_vector<int> posz(Nmax), which_vortex_is(Nmax), where_vortex_is(Nmax);

    // Cell list vectors
    thrust::device_vector<int> cell_ids(Nmax, -1);
    thrust::device_vector<int> cell_starts(Nc), cell_ends(Nc);
    thrust::device_vector<int> cell_range(Nc);
    thrust::sequence(cell_range.begin(), cell_range.end(), 0);

    // Positions as of the last cell-list rebuild, for the Verlet-skin drift check
    thrust::device_vector<float> posx_last(Nmax), posy_last(Nmax);
#if USE_CELL_LIST == 1
    int rebuild_count = 0;
#endif

    // Initialize position state. which_vortex_is[slot] = globally unique id.
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(Nmax),
        thrust::make_zip_iterator(thrust::make_tuple(
            posx.begin(), posy.begin(), posz.begin(), which_vortex_is.begin())),
        InitTriangularLattice{Nx, Nxy, a0, Lx, Ly}
    );

    // where_vortex_is[global id] = slot. Initially the identity permutation.
    thrust::sequence(where_vortex_is.begin(), where_vortex_is.end(), 0);

#if USE_CELL_LIST == 1
    std::cout << "Running with [Cell List O(N)] mode (Verlet skin = " << skin << ")...\n";

    // Initial build, so the first step's force evaluation has a valid list.
    rebuildCellList(posx, posy, posz, which_vortex_is, where_vortex_is,
                    cell_ids, cell_starts, cell_ends, cell_range,
                    Ncx, Ncy, Ncxy, cut_offx, cut_offy, Lx, Ly, Nmax);
    posx_last = posx;
    posy_last = posy;
    rebuild_count = 1;
#else
    std::cout << "Running with [Direct O(N^2)] mode...\n";
#endif

    printConfiguration("configIni.dat", 0, T, Nmax, Nxy,
                       posx, posy, posz, fx, fy, which_vortex_is, cell_ids);

    
    std::cout << "Starting simulation..." << std::endl;

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // --- Dynamic Loop ---
    for (int step = 0; step < steps; ++step) {

#if USE_CELL_LIST == 1
        // 1-4. Verlet-skin check: only re-sort into cells once some particle has
        // drifted more than skin/2 since the last rebuild. The mesh was sized to
        // cutoff+skin, so up to that drift the stale cell assignment is still a
        // superset of the true cutoff neighbor list (see the Verlet-skin section).
        float max_disp2 = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(
                posx.begin(), posy.begin(), posx_last.begin(), posy_last.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                posx.end(), posy.end(), posx_last.end(), posy_last.end())),
            DisplacementSq{Lx, Ly},
            0.0f,
            thrust::maximum<float>()
        );

        if (max_disp2 > skin_half_sq) {
            rebuildCellList(posx, posy, posz, which_vortex_is, where_vortex_is,
                            cell_ids, cell_starts, cell_ends, cell_range,
                            Ncx, Ncy, Ncxy, cut_offx, cut_offy, Lx, Ly, Nmax);
            posx_last = posx;
            posy_last = posy;
            ++rebuild_count;
        }

        // 5. Evaluate forces using the cell list
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(Nmax),
            thrust::make_zip_iterator(thrust::make_tuple(fx.begin(), fy.begin())),
            CalculateForcesCellList{
                thrust::raw_pointer_cast(posx.data()),
                thrust::raw_pointer_cast(posy.data()),
                thrust::raw_pointer_cast(posz.data()),
                thrust::raw_pointer_cast(cell_starts.data()),
                thrust::raw_pointer_cast(cell_ends.data()),
                thrust::raw_pointer_cast(which_vortex_is.data()),
                thrust::raw_pointer_cast(where_vortex_is.data()),
                Nmax, Nxy, Nz, Ncx, Ncy, Ncxy,
                Lx, Ly, k_spring, cutoff * cutoff, cut_offx, cut_offy
            }
        );
#else
        // Evaluate forces directly, O(N^2)
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(Nmax),
            thrust::make_zip_iterator(thrust::make_tuple(fx.begin(), fy.begin())),
            CalculateForcesDirect{
                thrust::raw_pointer_cast(posx.data()),
                thrust::raw_pointer_cast(posy.data()),
                thrust::raw_pointer_cast(posz.data()),
                thrust::raw_pointer_cast(which_vortex_is.data()),
                thrust::raw_pointer_cast(where_vortex_is.data()),
                Nmax, Nxy, Nz, Lx, Ly, k_spring, cutoff * cutoff
            }
        );
#endif
        // Advance positions via Langevin integration (folds back into the box)
        auto input_zip = thrust::make_zip_iterator(thrust::make_tuple(
            posx.begin(), posy.begin(), fx.begin(), fy.begin(), thrust::make_counting_iterator(0)
        ));
        auto output_zip = thrust::make_zip_iterator(thrust::make_tuple(posx.begin(), posy.begin()));

        thrust::transform(
            input_zip, input_zip + Nmax, output_zip,
            LangevinStep{
                thrust::raw_pointer_cast(which_vortex_is.data()),
                dt,
                s2Tdt,
                Lx,
                Ly,
                p.seed,
                static_cast<unsigned long long>(step)
            }
        );

        if (step % p.print_interval == 0 || step == steps - 1) {
            std::string out_name = "config_step_" + std::to_string(step) + ".dat";
            printConfiguration(out_name, step, T, Nmax, Nxy,
                               posx, posy, posz, fx, fy, which_vortex_is, cell_ids);
        }
    }

    // Report the final position of vortex 0 by identity, not by memory slot.
    thrust::host_vector<float> h_posx = posx;
    thrust::host_vector<float> h_posy = posy;
    thrust::host_vector<int>   h_where = where_vortex_is;
    int slot0 = h_where[0];
    std::cout << "Simulation complete! Final position Vortex 0: x = "
              << h_posx[slot0] << ", y = " << h_posy[slot0] << std::endl;

    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // Calculate durations
    std::chrono::duration<double> total_seconds = end_time - start_time;
    double ms_per_step = (total_seconds.count() * 1000.0) / steps;
    
    // Print timing results
    std::cout << "\n====================================================\n";
    std::cout << "               PERFORMANCE METRICS                  \n";
    std::cout << "====================================================\n";
    std::cout << "Total Run Time   : " << total_seconds.count() << " seconds\n";
    std::cout << "Average Time/Step: " << ms_per_step << " ms\n";
#if USE_CELL_LIST == 1
    std::cout << "Cell List Rebuilds: " << rebuild_count << " / " << steps
               << " steps (avg every " << (double(steps) / rebuild_count) << " steps)\n";
#endif
    std::cout << "====================================================\n";

    return 0;
}
