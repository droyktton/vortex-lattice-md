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
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
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
// Quenched point disorder (pinning centers)
//
// Each layer is divided into an implicit Npx x Npy grid of pinning cells, each
// at least the pin interaction range (3 r_p) wide. The pins in cell (cx, cy)
// of layer z are a pure function of (cx, cy, z, disorder_seed) through Philox:
// counter word 3 = 0 draws the Poisson pin count, = k >= 1 the position of pin
// k. Nothing is stored or rebuilt, and a restart sees the same pins given the
// same --disorder-seed. Dropping z from the counter (correlated = true) gives
// identical pins in every layer, i.e. columnar defects.
//
// Pin potential U = -Ap exp(-r^2/rp^2): attractive for Ap > 0.
// ============================================================================

struct PinningParams {
    bool enabled;              // false = no pinning at all (bit-identical to no disorder)
    bool correlated;
    int Npx, Npy;
    float lpx, lpy;            // pinning cell size
    float Ap, rp2, rcut2;
    float lambda;              // mean pins per pinning cell
    unsigned long long dseed;
};

#define PIN_MAX_PER_CELL 64

// Uniform in (0, 1) from a 32-bit Philox word.
__host__ __device__ inline float u01_open(uint32_t r) {
    return (float(r) + 0.5f) * 2.3283064365386963e-10f;
}

__host__ __device__ inline r123::Philox4x32::ctr_type pin_draw(
    const PinningParams& pp, int cx, int cy, int z, int k)
{
    // Key differs from the thermal-noise key in LangevinStep.
    r123::Philox4x32::key_type key = {{
        static_cast<uint32_t>(pp.dseed & 0xFFFFFFFFull),
        static_cast<uint32_t>(pp.dseed >> 32) ^ 0x85EBCA6Bu
    }};
    r123::Philox4x32::ctr_type ctr = {{
        static_cast<uint32_t>(cx),
        static_cast<uint32_t>(cy),
        static_cast<uint32_t>(pp.correlated ? 0 : z),
        static_cast<uint32_t>(k)
    }};
    r123::Philox4x32 gen;
    return gen(ctr, key);
}

// Number of pins in a pinning cell: Poisson(lambda), by inversion.
__host__ __device__ inline int pin_count(const PinningParams& pp, int cx, int cy, int z) {
    float u = u01_open(pin_draw(pp, cx, cy, z, 0)[0]);
    float p = expf(-pp.lambda);
    float cdf = p;
    int n = 0;
    while (u > cdf && n < PIN_MAX_PER_CELL) {
        ++n;
        p *= pp.lambda / n;
        cdf += p;
    }
    return n;
}

// Position of pin k (0-based) in a pinning cell.
__host__ __device__ inline void pin_position(
    const PinningParams& pp, int cx, int cy, int z, int k, float& px, float& py)
{
    r123::Philox4x32::ctr_type r = pin_draw(pp, cx, cy, z, k + 1);
    px = (cx + u01_open(r[0])) * pp.lpx;
    py = (cy + u01_open(r[1])) * pp.lpy;
}

__host__ __device__ inline void add_pinning_forces(
    float x, float y, int z, float Lx, float Ly,
    const PinningParams& pp, float& fx, float& fy)
{
    if (!pp.enabled) return;

    int cx = int(fold(x, Lx) / pp.lpx);
    int cy = int(fold(y, Ly) / pp.lpy);
    if (cx >= pp.Npx) cx = pp.Npx - 1;
    if (cy >= pp.Npy) cy = pp.Npy - 1;

    float pref = -2.0f * pp.Ap / pp.rp2;

    // Npx, Npy >= 3 is enforced in main(), so the 3x3 cells are distinct.
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int ncx = (cx + dx + pp.Npx) % pp.Npx;
            int ncy = (cy + dy + pp.Npy) % pp.Npy;
            int n = pin_count(pp, ncx, ncy, z);
            for (int k = 0; k < n; ++k) {
                float px, py;
                pin_position(pp, ncx, ncy, z, k, px, py);
                float rx = min_dist(x - px, Lx);
                float ry = min_dist(y - py, Ly);
                float r2 = rx * rx + ry * ry;
                if (r2 < pp.rcut2) {
                    float f = pref * expf(-r2 / pp.rp2);
                    fx += f * rx;
                    fy += f * ry;
                }
            }
        }
    }
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
    PinningParams pin;

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

        add_pinning_forces(x1, y1, z1, Lx, Ly, pin, fx, fy);

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
    PinningParams pin;

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

        add_pinning_forces(x1, y1, z1, Lx, Ly, pin, fx, fy);

        return thrust::make_tuple(fx, fy);
    }
};

// ============================================================================
// Integrator
// ============================================================================

// Overdamped Langevin (Brownian) step, with the updated position folded back
// into the periodic box.
// ============================================================================
// External driving force (optional)
//
// A uniform force on every vortex, along a fixed direction e = (cos a, sin a):
//     F(t) = [Fdc + Fac sin(2 pi f t)] e,     t = step * dt.
// It enters only the Langevin update (it is not part of fx/fy, so snapshots
// keep holding the interaction + pinning forces). Off unless Fdc or Fac is
// nonzero; when off, the update is literally the undriven one.
//
// Velocities: the dynamics is overdamped with unit mobility, so a vortex's
// mean velocity over the noise is its deterministic force. The velocity of
// layer z is therefore <fx, fy>_z + F(t), with no noise to average away. In-
// plane pair forces cancel within a layer, so <f>_z is the pinning force plus
// the net spring pull from the neighboring layers; averaged over all layers
// the springs cancel too, and the lattice velocity is F(t) + <f_pin>.
// ============================================================================

struct DriveParams {
    bool enabled;
    float Fdc, Fac, freq, angle_deg;
    float ex, ey;       // unit vector along the drive
};

// Drive amplitude at physical time t (double: t can get large)
inline double driveAmplitude(const DriveParams& d, double t) {
    return d.Fdc + d.Fac * sin(2.0 * M_PI * d.freq * t);
}

// (fx, fy) promoted to double, for the per-layer force sums
struct ToDoublePair {
    __host__ __device__
    thrust::tuple<double, double> operator()(const thrust::tuple<float, float>& t) const {
        return thrust::make_tuple(double(thrust::get<0>(t)), double(thrust::get<1>(t)));
    }
};

struct AddPair {
    __host__ __device__
    thrust::tuple<double, double> operator()(const thrust::tuple<double, double>& a,
                                             const thrust::tuple<double, double>& b) const {
        return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                  thrust::get<1>(a) + thrust::get<1>(b));
    }
};

struct LangevinStep {
    const int* which_vortex_is;
    float dt, s2Tdt, Lx, Ly;
    unsigned long long seed, step;
    bool driven;        // false: the undriven update, bit for bit
    float Fdx, Fdy;     // drive force this step

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

        if (driven) {
            x += (fx + Fdx) * dt + noise.x * s2Tdt;
            y += (fy + Fdy) * dt + noise.y * s2Tdt;
        } else {
            x += fx * dt + noise.x * s2Tdt;
            y += fy * dt + noise.y * s2Tdt;
        }

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

// Dump every pin as "x y z", one blank-line-separated block per layer (same
// layout as the config files, for gnuplot). Returns the pin count per layer.
// Uses the same pin_count/pin_position the force kernels evaluate.
std::vector<int> writePins(const std::string& filename, const PinningParams& pin, int Nz) {
    std::vector<int> counts(Nz, 0);
    std::ofstream fout(filename);
    if (!fout.is_open())
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";

    fout << "x\ty\tz\n";
    fout << std::fixed << std::setprecision(6);
    for (int z = 0; z < Nz; ++z) {
        for (int cy = 0; cy < pin.Npy; ++cy) {
            for (int cx = 0; cx < pin.Npx; ++cx) {
                int n = pin_count(pin, cx, cy, z);
                for (int k = 0; k < n; ++k) {
                    float px, py;
                    pin_position(pin, cx, cy, z, k, px, py);
                    fout << px << "\t" << py << "\t" << z << "\n";
                }
                counts[z] += n;
            }
        }
        fout << "\n\n";
    }
    return counts;
}

// Parameters of the optional quenched disorder, for the log. Only written
// when pinning is enabled, so logs of runs without disorder are unchanged.
struct DisorderLog {
    float np, rp, rcut, dt_max;
    std::vector<int> pins_per_layer;
};

void writeSimulationLog(
    const std::string& filename,
    int Nx, int Ny, int Nz, int Nmax,
    float a0, float Lx, float Ly,
    float k_spring, float T, float dt, int steps, float cutoff,
    int Ncx, int Ncy, int Ncxy, int Nc, float cut_offx, float cut_offy,
    bool use_cell_list, float skin, float cutoff_skin, unsigned long long seed,
    const PinningParams& pin, const DisorderLog& dis,
    const DriveParams& drive, int velocity_interval
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
    log << "Core Radius          : " << R_CORE << "\n";
    log << "RNG Seed             : " << seed << "\n\n";

    if (pin.enabled) {
        double expected = double(pin.lambda) * pin.Npx * pin.Npy;
        log << "[Quenched Disorder]\n";
        log << "Pin Density (np)     : " << dis.np << "  (pins per unit area, per layer)\n";
        log << "Pin Strength (Ap)    : " << pin.Ap << "  (U = -Ap exp(-r^2/rp^2))\n";
        log << "Pin Range (rp)       : " << dis.rp << "\n";
        log << "Pin Force Cutoff     : " << dis.rcut << "  (3 rp)\n";
        log << "Disorder Seed        : " << pin.dseed << "\n";
        log << "Disorder Correlated  : " << (pin.correlated ? 1 : 0)
            << "  (0 = independent per layer, 1 = identical in every layer / columnar)\n";
        log << "Pinning Grid         : " << pin.Npx << " x " << pin.Npy
            << "  (cell " << pin.lpx << " x " << pin.lpy << ")\n";
        log << "Pins per Layer       :";
        for (int n : dis.pins_per_layer) log << " " << n;
        log << "  (Poisson, mean " << expected << ")\n";
        log << "Suggested Max dt     : " << dis.dt_max << "\n\n";
    }

    if (drive.enabled) {
        log << "[Driving Force]\n";
        log << "Drive DC Force (Fdc) : " << drive.Fdc << "\n";
        log << "Drive AC Force (Fac) : " << drive.Fac << "  (F(t) = [Fdc + Fac sin(2 pi f t)] e)\n";
        log << "Drive Frequency (f)  : " << drive.freq << "\n";
        log << "Drive Angle (deg)    : " << drive.angle_deg << "  (e = (cos, sin); 0 = along x)\n";
        log << "Velocity Interval    : " << velocity_interval << "  (steps between velocity.dat rows)\n\n";
    }

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
// Restart / checkpoint
//
// The Philox RNG in LangevinStep is stateless -- keyed on (gid, step, seed),
// not on any running state -- so resuming a run needs no RNG state at all,
// just the physical state (positions, layers, identities) and the right
// step number to continue counting from. A config_step_<N>.dat already has
// everything needed for the former; the loop below supplies the latter.
// ============================================================================

// Extract N from a "config_step_<N>.dat" path (any directory prefix is
// ignored), matching the same convention the Python tooling uses.
bool parseStepFromFilename(const std::string& path, int& step) {
    size_t slash = path.find_last_of("/\\");
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const std::string prefix = "config_step_", suffix = ".dat";
    if (base.size() <= prefix.size() + suffix.size()) return false;
    if (base.compare(0, prefix.size(), prefix) != 0) return false;
    if (base.compare(base.size() - suffix.size(), suffix.size(), suffix) != 0) return false;
    std::string digits = base.substr(prefix.size(), base.size() - prefix.size() - suffix.size());
    if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) return false;
    step = std::stoi(digits);
    return true;
}

std::string baseName(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    return (slash == std::string::npos) ? path : path.substr(slash + 1);
}

std::string dirName(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    return (slash == std::string::npos) ? std::string(".") : path.substr(0, slash);
}

// Read a "label : value" line from a simulation.log, mirroring the Python
// tooling's find_log_value(). Returns false if the file or label isn't found.
bool findLogValue(const std::string& log_path, const std::string& label, double& value) {
    std::ifstream in(log_path);
    if (!in.is_open()) return false;
    std::string line;
    while (std::getline(in, line)) {
        size_t pos = line.find(label);
        if (pos == std::string::npos) continue;
        size_t colon = line.find(':', pos);
        if (colon == std::string::npos) continue;
        try {
            value = std::stod(line.substr(colon + 1));
        } catch (...) {
            return false;
        }
        return true;
    }
    return false;
}

// Best-effort cross-check: nothing about --nx/--ny/--nz/--a0/--k/--T/--dt/
// --cutoff/--skin/--seed is recoverable from the restart file's physical
// state alone, so a mismatched value silently continues the trajectory
// under different physics (or, for the seed, replays noise already used).
// This warns (doesn't block) when the original run's own simulation.log
// disagrees with the current invocation.
void checkRestartConsistency(
    const std::string& log_path, int Nmax, float a0, float k_spring,
    float T, float dt, float cutoff, float skin, unsigned long long seed,
    const PinningParams& pin, float np, float rp, const DriveParams& drive)
{
    std::ifstream test(log_path);
    if (!test.is_open()) {
        std::cerr << "Warning: no simulation.log found at " << log_path << "; cannot cross-check "
                     "run parameters (--nx/--ny/--nz/--a0/--k/--T/--dt/--cutoff/--skin/--seed) "
                     "against the original run.\n";
        return;
    }

    struct Check { const char* label; double current; };
    Check checks[] = {
        {"Total Vortices (Nmax)", double(Nmax)},
        {"Lattice Constant (a0)", double(a0)},
        {"Spring Constant (k)",   double(k_spring)},
        {"Temperature (T)",       double(T)},
        {"Time Step (dt)",        double(dt)},
        {"Interaction Cutoff",    double(cutoff)},
        {"Skin",                  double(skin)},
        {"RNG Seed",              double(seed)},
        // Disorder labels are absent from logs of runs without disorder, and
        // then skipped below like any other missing label.
        {"Pin Density (np)",      pin.enabled ? double(np) : 0.0},
        {"Pin Strength (Ap)",     pin.enabled ? double(pin.Ap) : 0.0},
        // Only meaningful if this invocation has disorder too.
        {pin.enabled ? "Pin Range (rp)" : nullptr,      double(rp)},
        {pin.enabled ? "Disorder Seed" : nullptr,       double(pin.dseed)},
        {pin.enabled ? "Disorder Correlated" : nullptr, pin.correlated ? 1.0 : 0.0},
        // Same for the drive: absent from undriven logs.
        {"Drive DC Force (Fdc)", drive.enabled ? double(drive.Fdc) : 0.0},
        {"Drive AC Force (Fac)", drive.enabled ? double(drive.Fac) : 0.0},
        {drive.enabled ? "Drive Frequency (f)" : nullptr, double(drive.freq)},
        {drive.enabled ? "Drive Angle (deg)" : nullptr,   double(drive.angle_deg)},
    };

    for (const auto& c : checks) {
        if (c.label == nullptr) continue;
        double logged;
        if (!findLogValue(log_path, c.label, logged)) continue;  // e.g. Skin absent in direct mode
        double denom = std::fabs(logged) > 1e-12 ? std::fabs(logged) : 1.0;
        if (std::fabs(logged - c.current) / denom > 1e-4) {
            std::cerr << "Warning: " << c.label << " mismatch with the original run -- "
                      << "log says " << logged << ", this invocation says " << c.current
                      << ". Restarting with different physical parameters changes the "
                         "trajectory you're continuing.\n";
        }
    }
}

// Load position/layer/identity state from a previously printed
// config_step_N.dat (or configIni.dat) to resume a run. Forces aren't read
// back -- they get recomputed fresh before the first restarted step anyway.
bool loadConfiguration(
    const std::string& filename, int Nmax,
    thrust::device_vector<float>& posx,
    thrust::device_vector<float>& posy,
    thrust::device_vector<int>& posz,
    thrust::device_vector<int>& which_vortex_is)
{
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error: could not open restart file " << filename << "\n";
        return false;
    }

    std::string line;
    std::getline(fin, line);  // "Iter = ..."
    std::getline(fin, line);  // "Temperatura = ..."
    std::getline(fin, line);  // column header

    // Read every row the file has, not just the first Nmax -- capping the
    // read at Nmax would silently accept a file with MORE rows than
    // expected (e.g. from a mismatched --nx/--ny/--nz), truncating it
    // instead of catching the mismatch.
    std::vector<float> vx, vy;
    std::vector<int> vz, vgid;
    vx.reserve(Nmax); vy.reserve(Nmax); vz.reserve(Nmax); vgid.reserve(Nmax);

    float x, y, fxv, fyv;
    int z, gid, label, cell;
    // operator>> skips all whitespace, including the blank lines
    // printConfiguration() inserts between layers.
    while (fin >> x >> y >> z >> fxv >> fyv >> gid >> label >> cell) {
        vx.push_back(x);
        vy.push_back(y);
        vz.push_back(z);
        vgid.push_back(gid);
    }

    if (static_cast<int>(vx.size()) != Nmax) {
        std::cerr << "Error: restart file " << filename << " has " << vx.size()
                  << " rows, expected " << Nmax << " (check --nx/--ny/--nz match the original run).\n";
        return false;
    }

    thrust::host_vector<float> h_posx(vx.begin(), vx.end());
    thrust::host_vector<float> h_posy(vy.begin(), vy.end());
    thrust::host_vector<int> h_posz(vz.begin(), vz.end());
    thrust::host_vector<int> h_which(vgid.begin(), vgid.end());

    posx = h_posx;
    posy = h_posy;
    posz = h_posz;
    which_vortex_is = h_which;
    return true;
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
    std::string restart_file;   // empty = fresh start
    int start_step = -1;        // -1 = auto-detect from the restart filename
    // Quenched disorder: off unless both np > 0 and Ap != 0.
    float np = 0.0f;
    float Ap = 0.0f;
    float rp = 0.1f;
    unsigned long long disorder_seed = 1ULL;
    bool correlated_disorder = false;
    // External drive: off unless Fdc or Fac is nonzero.
    float Fdc = 0.0f;
    float Fac = 0.0f;
    float freq = 0.0f;
    float drive_angle = 0.0f;   // degrees
    int velocity_interval = -1; // -1 = same as print_interval
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
        "  --restart FILE      resume from a config_step_<N>.dat (or configIni.dat)\n"
        "                      instead of a fresh lattice; pass the SAME --nx/--ny/--nz/\n"
        "                      --a0/--k/--T/--dt/--cutoff/--skin/--seed as the original run\n"
        "  --start-step N      step to resume at (default: parsed from --restart's filename)\n"
        "\n"
        "Quenched disorder (off by default; enabled when --np > 0 and --Ap != 0):\n"
        "  --np F              pinning centers per unit area, per layer (default 0)\n"
        "  --Ap F              pin strength, U = -Ap exp(-r^2/rp^2)    (default 0)\n"
        "  --rp F              pin range                              (default 0.1)\n"
        "  --disorder-seed N   seed of the disorder realization        (default 1)\n"
        "  --correlated-disorder  same pins in every layer (columnar) instead of\n"
        "                      independent pins per layer\n"
        "\n"
        "External drive (off by default; enabled when --Fdc or --Fac is nonzero):\n"
        "  F(t) = [Fdc + Fac sin(2 pi freq t)] (cos angle, sin angle), on every vortex\n"
        "  --Fdc F             DC force                               (default 0)\n"
        "  --Fac F             AC force amplitude                     (default 0)\n"
        "  --freq F            AC frequency, in 1/time units          (default 0)\n"
        "  --drive-angle F     drive direction, degrees from x        (default 0)\n"
        "  --velocity-interval N  steps between velocity.dat rows (default: --print-interval)\n"
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
            else if (arg == "--restart") p.restart_file = next();
            else if (arg == "--start-step") p.start_step = std::stoi(next());
            else if (arg == "--np") p.np = std::stof(next());
            else if (arg == "--Ap") p.Ap = std::stof(next());
            else if (arg == "--rp") p.rp = std::stof(next());
            else if (arg == "--disorder-seed") p.disorder_seed = std::stoull(next());
            else if (arg == "--correlated-disorder") p.correlated_disorder = true;
            else if (arg == "--Fdc") p.Fdc = std::stof(next());
            else if (arg == "--Fac") p.Fac = std::stof(next());
            else if (arg == "--freq") p.freq = std::stof(next());
            else if (arg == "--drive-angle") p.drive_angle = std::stof(next());
            else if (arg == "--velocity-interval") p.velocity_interval = std::stoi(next());
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

    // --- Quenched disorder (see the pinning section above) ---
    PinningParams pin{};          // all zero: disabled
    DisorderLog dis{};
    pin.enabled = (p.np > 0.0f && p.Ap != 0.0f);
    if (!pin.enabled && (p.np > 0.0f || p.Ap != 0.0f)) {
        std::cerr << "Warning: quenched disorder needs both --np > 0 and --Ap != 0; "
                     "running without disorder.\n";
    }
    if (pin.enabled) {
        if (p.rp <= 0.0f) {
            std::cerr << "Error: --rp must be positive.\n";
            return EXIT_FAILURE;
        }
        float rcut_p = 3.0f * p.rp;
        pin.correlated = p.correlated_disorder;
        pin.Npx = int(Lx / rcut_p);
        pin.Npy = int(Ly / rcut_p);
        if (pin.Npx < 3 || pin.Npy < 3) {
            std::cerr << "Error: the box must be at least 3 pin ranges (9 rp) wide in x and y; "
                         "reduce --rp or enlarge the system.\n";
            return EXIT_FAILURE;
        }
        pin.lpx = Lx / pin.Npx;
        pin.lpy = Ly / pin.Npy;
        pin.Ap = p.Ap;
        pin.rp2 = p.rp * p.rp;
        pin.rcut2 = rcut_p * rcut_p;
        pin.lambda = p.np * pin.lpx * pin.lpy;
        pin.dseed = p.disorder_seed;
        // The Poisson draw is capped at PIN_MAX_PER_CELL pins per cell.
        if (pin.lambda > 16.0f) {
            std::cerr << "Error: " << pin.lambda << " pins per pinning cell on average (np * (3 rp)^2"
                         " or more); this is beyond the supported density. Reduce --np or --rp.\n";
            return EXIT_FAILURE;
        }

        // A pin's force peaks at Fmax = sqrt(2/e) Ap/rp, at r = rp/sqrt(2). Resolving
        // the well needs Fmax*dt <= 0.1 rp and sqrt(2 T dt) <= rp, both ~ rp^2.
        // Only warn: dt is never changed behind the user's back.
        float fmax = 0.8577639f * fabsf(p.Ap) / p.rp;
        float dt_max = 0.1f * p.rp / fmax;
        if (T > 0.0f) dt_max = fminf(dt_max, pin.rp2 / (2.0f * T));
        if (dt > dt_max) {
            std::cerr << "Warning: --dt " << dt << " is too large to resolve the pinning wells "
                         "(rp = " << p.rp << ", Ap = " << p.Ap << ", T = " << T
                      << "); suggested dt <= " << dt_max << ".\n";
        }

        dis.np = p.np;
        dis.rp = p.rp;
        dis.rcut = rcut_p;
        dis.dt_max = dt_max;
        dis.pins_per_layer = writePins("pins.dat", pin, Nz);
        std::cout << "Quenched disorder: " << (pin.correlated ? "columnar" : "uncorrelated")
                  << " pinning, pins written to pins.dat\n";
    }

    // --- External drive (see the driving-force section above) ---
    DriveParams drive{};          // all zero: disabled
    drive.enabled = (p.Fdc != 0.0f || p.Fac != 0.0f);
    int velocity_interval = p.velocity_interval > 0 ? p.velocity_interval : p.print_interval;
    if (drive.enabled) {
        if (p.Fac != 0.0f && p.freq <= 0.0f) {
            std::cerr << "Error: --Fac needs a positive --freq.\n";
            return EXIT_FAILURE;
        }
        drive.Fdc = p.Fdc;
        drive.Fac = p.Fac;
        drive.freq = p.freq;
        drive.angle_deg = p.drive_angle;
        double a = p.drive_angle * M_PI / 180.0;
        drive.ex = float(cos(a));
        drive.ey = float(sin(a));

        // Warnings only, like the pinning dt check.
        float fmax_drive = fabsf(p.Fdc) + fabsf(p.Fac);
        float lmin = pin.enabled ? fminf(a0, p.rp) : a0;
        if (fmax_drive * dt > 0.1f * lmin) {
            std::cerr << "Warning: the drive moves vortices " << fmax_drive * dt << " per step, more than "
                         "0.1 of the smallest length scale (" << lmin << "); suggested dt <= "
                      << 0.1f * lmin / fmax_drive << ".\n";
        }
        if (fmax_drive * dt * p.print_interval > 0.25f * fminf(Lx, Ly)) {
            std::cerr << "Warning: vortices drift " << fmax_drive * dt * p.print_interval
                      << " between snapshots, over a quarter of the box; msd.py cannot unwrap "
                         "trajectories that coarse. Reduce --print-interval.\n";
        }
        if (p.Fac != 0.0f) {
            double period = 1.0 / p.freq;
            if (period < 20.0 * dt)
                std::cerr << "Warning: the AC period (" << period << ") is under 20 timesteps.\n";
            if (period < 8.0 * velocity_interval * dt)
                std::cerr << "Warning: velocity.dat samples the AC period (" << period
                          << ") fewer than 8 times; reduce --velocity-interval.\n";
        }
        std::cout << "External drive: Fdc = " << p.Fdc << ", Fac = " << p.Fac << ", f = " << p.freq
                  << ", angle = " << p.drive_angle << " deg; velocities written to velocity.dat\n";
    }

    // --- Restart bookkeeping ---
    bool restarting = !p.restart_file.empty();
    int start_step = 0;
    if (restarting) {
        int parsed_step;
        if (p.start_step >= 0) {
            start_step = p.start_step;
        } else if (baseName(p.restart_file) == "configIni.dat") {
            start_step = 0;
        } else if (parseStepFromFilename(p.restart_file, parsed_step)) {
            start_step = parsed_step + 1;
        } else {
            std::cerr << "Error: could not determine the step to resume from for '"
                      << p.restart_file << "'. Pass --start-step explicitly.\n";
            return EXIT_FAILURE;
        }

        if (start_step >= steps) {
            std::cerr << "Error: --start-step " << start_step << " is not before --steps "
                      << steps << "; nothing to do.\n";
            return EXIT_FAILURE;
        }

        std::cout << "Restarting from " << p.restart_file << " at step " << start_step << "\n";
        checkRestartConsistency(dirName(p.restart_file) + "/simulation.log",
                                Nmax, a0, k_spring, T, dt, cutoff, skin, p.seed,
                                pin, p.np, p.rp, drive);
    }

    writeSimulationLog(
        "simulation.log",
        Nx, Ny, Nz, Nmax,
        a0, Lx, Ly,
        k_spring, T, dt, steps, cutoff,
        Ncx, Ncy, Ncxy, Nc, cut_offx, cut_offy,
        use_cell_list, skin, cutoff_skin, p.seed,
        pin, dis, drive, velocity_interval
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
    if (restarting) {
        if (!loadConfiguration(p.restart_file, Nmax, posx, posy, posz, which_vortex_is))
            return EXIT_FAILURE;
    } else {
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(Nmax),
            thrust::make_zip_iterator(thrust::make_tuple(
                posx.begin(), posy.begin(), posz.begin(), which_vortex_is.begin())),
            InitTriangularLattice{Nx, Nxy, a0, Lx, Ly}
        );
    }

    // where_vortex_is[global id] = slot. Valid whether which_vortex_is is the
    // identity permutation (fresh start) or loaded from a restart file.
    thrust::scatter(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(Nmax),
        which_vortex_is.begin(),
        where_vortex_is.begin()
    );

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

    // configIni.dat represents the pristine initial lattice, so don't
    // overwrite it when continuing an already-running trajectory.
    if (!restarting) {
        printConfiguration("configIni.dat", 0, T, Nmax, Nxy,
                           posx, posy, posz, fx, fy, which_vortex_is, cell_ids);
    }

    
    // velocity.dat: one row every velocity_interval steps (and the last step).
    // A restart in the same folder appends to it instead of starting over.
    std::ofstream vel_out;
    std::vector<double> vel_sum(2, 0.0);
    long vel_samples = 0;
    // reduce_by_key outputs; at most one segment per vortex (Nmax), one per
    // layer in practice. Only allocated when driven.
    thrust::device_vector<int> seg_keys(drive.enabled ? Nmax : 0);
    thrust::device_vector<double> seg_fx(drive.enabled ? Nmax : 0), seg_fy(drive.enabled ? Nmax : 0);
    if (drive.enabled) {
        std::ifstream existing("velocity.dat");
        bool append = restarting && existing.good();
        existing.close();
        vel_out.open("velocity.dat", append ? std::ios::app : std::ios::trunc);
        if (!append) {
            vel_out << "step\tt\tFx\tFy\tvx\tvy";
            for (int z = 0; z < Nz; ++z) vel_out << "\tvx_z" << z << "\tvy_z" << z;
            vel_out << "\n";
        }
        vel_out << std::setprecision(8);
    }

    std::cout << "Starting simulation..." << std::endl;

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // --- Dynamic Loop ---
    for (int step = start_step; step < steps; ++step) {

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
                Lx, Ly, k_spring, cutoff * cutoff, cut_offx, cut_offy,
                pin
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
                Nmax, Nxy, Nz, Lx, Ly, k_spring, cutoff * cutoff,
                pin
            }
        );
#endif
        // Drive force this step, evaluated at the start of the step (explicit Euler)
        float Fdx = 0.0f, Fdy = 0.0f;
        if (drive.enabled) {
            double amp = driveAmplitude(drive, double(step) * dt);
            Fdx = float(amp * drive.ex);
            Fdy = float(amp * drive.ey);
        }

        // Velocity of every layer: <f>_z + F(t) (see the driving-force section)
        if (drive.enabled && (step % velocity_interval == 0 || step == steps - 1)) {
            // One pass: memory is always grouped by layer (layer-major at the
            // start, and the cell list sorts by z*Ncxy + cell), so reduce_by_key
            // on posz gives one segment per layer. Segments are still summed by
            // key on the host, so the result is right whatever the order.
            auto f_begin = thrust::make_transform_iterator(
                thrust::make_zip_iterator(thrust::make_tuple(fx.begin(), fy.begin())), ToDoublePair());
            auto seg_end = thrust::reduce_by_key(
                posz.begin(), posz.end(), f_begin,
                seg_keys.begin(),
                thrust::make_zip_iterator(thrust::make_tuple(seg_fx.begin(), seg_fy.begin())),
                thrust::equal_to<int>(), AddPair());
            int nseg = int(seg_end.first - seg_keys.begin());
            thrust::host_vector<int> h_keys(seg_keys.begin(), seg_keys.begin() + nseg);
            thrust::host_vector<double> h_sfx(seg_fx.begin(), seg_fx.begin() + nseg);
            thrust::host_vector<double> h_sfy(seg_fy.begin(), seg_fy.begin() + nseg);
            std::vector<double> vx(Nz, 0.0), vy(Nz, 0.0);
            for (int k = 0; k < nseg; ++k) {
                vx[h_keys[k]] += h_sfx[k];
                vy[h_keys[k]] += h_sfy[k];
            }
            double vx_all = 0.0, vy_all = 0.0;
            for (int z = 0; z < Nz; ++z) {
                vx[z] = vx[z] / Nxy + Fdx;
                vy[z] = vy[z] / Nxy + Fdy;
                vx_all += vx[z] / Nz;
                vy_all += vy[z] / Nz;
            }
            vel_out << step << "\t" << double(step) * dt << "\t" << Fdx << "\t" << Fdy
                    << "\t" << vx_all << "\t" << vy_all;
            for (int z = 0; z < Nz; ++z) vel_out << "\t" << vx[z] << "\t" << vy[z];
            vel_out << "\n";
            vel_sum[0] += vx_all;
            vel_sum[1] += vy_all;
            ++vel_samples;
        }

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
                static_cast<unsigned long long>(step),
                drive.enabled,
                Fdx,
                Fdy
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
    if (drive.enabled && vel_samples > 0) {
        std::cout << "Mean lattice velocity over this run (" << vel_samples << " samples): vx = "
                  << vel_sum[0] / vel_samples << ", vy = " << vel_sum[1] / vel_samples << "\n";
    }

    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // Calculate durations. Divide by the steps actually executed THIS
    // invocation, not the final target `steps` -- on a restart those differ.
    int executed_steps = steps - start_step;
    std::chrono::duration<double> total_seconds = end_time - start_time;
    double ms_per_step = (total_seconds.count() * 1000.0) / executed_steps;

    // Print timing results
    std::cout << "\n====================================================\n";
    std::cout << "               PERFORMANCE METRICS                  \n";
    std::cout << "====================================================\n";
    std::cout << "Total Run Time   : " << total_seconds.count() << " seconds\n";
    std::cout << "Average Time/Step: " << ms_per_step << " ms\n";
#if USE_CELL_LIST == 1
    std::cout << "Cell List Rebuilds: " << rebuild_count << " / " << executed_steps
               << " steps (avg every " << (double(executed_steps) / rebuild_count) << " steps)\n";
#endif
    std::cout << "====================================================\n";

    return 0;
}
