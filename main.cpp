#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "polyscope/pick.h"
#include <igl/jet.h>
#include <Eigen/Sparse>

struct SimulationData
{
    // Created for you when you initialize the grid. You shouldn't modify these yourself.
    int gridn; // The grid has gridn x gridn vertices    
    double gridh; // The width and height of each grid cell
    Eigen::MatrixXd V; // The grid vertex positions (used for rendering)
    Eigen::MatrixXi F; // The grid face indices (used for rendering)
    int nparticles; // number of marker particles to maintain

    // Simulation parameters
    double timestep;
    double density;
    double viscosity;
    double mouseStrength;
    
    // Configuration vectors. Update these in your code.
    Eigen::MatrixXd u; // velocity field, size  gridn * gridn x 2
    Eigen::VectorXd p; // pressure field, size (gridn-1)*(gridn-1)
    Eigen::MatrixXd markerParticles; // locations of the maker particles. Size nparticles x 3, z coordinate always zero.

    // Computed for you from the GUI
    Eigen::MatrixXd externalForce; // size gridn * gridn x 2



    Eigen::SparseMatrix<double> M_inv;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> vertSolver;
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> cellSolver;
};

// Create external force based on mouse drag

void setExternalForce(SimulationData& simdata, double x, double y, double vx, double vy)
{
    simdata.externalForce.setZero();
    // find where mouse is
    int i = int((x + 1.0) / simdata.gridh);
    int j = int((y + 1.0) / simdata.gridh);
    if (i >= 0 && j >= 0 && i < simdata.gridn - 1 && j < simdata.gridn - 1)
    {
        double u = (x - (-1.0 + i * simdata.gridh)) / simdata.gridh;
        double v = (y - (-1.0 + j * simdata.gridh)) / simdata.gridh;
        Eigen::Vector2d vec(vx, vy);
        vec *= simdata.mouseStrength;
        simdata.externalForce.row(j * simdata.gridn + i) = (1.0 - u) * (1.0 - v) * vec;
        simdata.externalForce.row((j + 1) * simdata.gridn + i) = (1.0 - u) * v * vec;
        simdata.externalForce.row(j * simdata.gridn + i + 1) = u * (1.0 - v) * vec;
        simdata.externalForce.row((j + 1) * simdata.gridn + i + 1) = u * v * vec;
    }
}

// Convert pressure to colors for the GUI

void makeFaceColors(SimulationData& simdata, Eigen::VectorXd& colors)
{
    colors.resize(2 * (simdata.gridn - 1) * (simdata.gridn - 1));
    for (int i = 0; i < simdata.gridn - 1; i++)
    {
        for (int j = 0; j < simdata.gridn - 1; j++)
        {
            colors[2 * (j * (simdata.gridn - 1) + i) + 0] = simdata.p[j * (simdata.gridn - 1) + i];
            colors[2 * (j * (simdata.gridn - 1) + i) + 1] = simdata.p[j * (simdata.gridn - 1) + i];
        }
    }
}

bool isCorner(int i, int j, int n) {
    return 
        (i == 0 && j == 0) || 
        (i == n - 1 && j == 0) || 
        (i == 0 && j == n - 1) || 
        (i == n - 1 && j == n - 1);
}

bool isEdge(int i, int j, int n) {
    return 
        i == 0 || 
        i == n - 1 ||
        j == 0 || 
        j == n - 1; 
}

bool isEdge(int i, int n) {
    return i == 0 || i == n - 1;
}

void updateMassInv(SimulationData &simdata) {
    int n = simdata.gridn;
    simdata.M_inv.resize(n * n, n * n);
    std::vector<Eigen::Triplet<double>> triplets;

    // loop through the vertices
    double m = simdata.gridh * simdata.gridh;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;

            // add the triplet at idx, idx
            if (isCorner(i, j, n))
                triplets.emplace_back(Eigen::Triplet<double>(idx, idx, m / 4));
            else if (isEdge(i, j, n))
                triplets.emplace_back(Eigen::Triplet<double>(idx, idx, m / 2));
            else
                triplets.emplace_back(Eigen::Triplet<double>(idx, idx, m));
        }
    }
    
    simdata.M_inv.setFromTriplets(triplets.begin(), triplets.end());
}

Eigen::SparseMatrix<double> getDMatrix(int n) {
    std::vector<Eigen::Triplet<double>> triplets;

    int edge = 0;
    // horizontal directs right
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - 1; ++j) {
            int vert = i * n + j;
            triplets.emplace_back(Eigen::Triplet<double>(edge, vert, -1));
            // pt directly to right
            triplets.emplace_back(Eigen::Triplet<double>(edge, vert + 1, 1));
            edge++;
        }
    }

    // vertical directs up
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n; ++j) {
            int vert = i * n + j;
            triplets.emplace_back(Eigen::Triplet<double>(edge, vert, 1));
            // pt directly below
            triplets.emplace_back(Eigen::Triplet<double>(edge, vert + n, -1));
            edge++;
        }
    }

    Eigen::SparseMatrix<double> d;
    d.resize(2 * n * (n - 1), n * n);
    d.setFromTriplets(triplets.begin(), triplets.end());
    return d;
}

Eigen::SparseMatrix<double> getStarMatrix(int n, int h) {
    std::vector<Eigen::Triplet<double>> triplets;
    int edge = 0;

    // horizontal; halve if one pt on edge
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - 1; ++j) {
            if (isEdge(i, n))
                triplets.emplace_back(Eigen::Triplet<double>(edge, edge, 0.5));
            else
                triplets.emplace_back(Eigen::Triplet<double>(edge, edge, 1));
            edge++;
        }
    }

    // vertical; halve if one pt on edge
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n; ++j) {
            if (isEdge(j, n))
                triplets.emplace_back(Eigen::Triplet<double>(edge, edge, 0.5));
            else
                triplets.emplace_back(Eigen::Triplet<double>(edge, edge, 1));
            edge++;
        }
    }

    Eigen::SparseMatrix<double> s;
    s.resize(2 * n * (n - 1), 2 * n * (n - 1));
    s.setFromTriplets(triplets.begin(), triplets.end());
    return s;
}

void computeVertLaplacian(SimulationData& simdata) {
    Eigen::SparseMatrix<double> vertD = getDMatrix(simdata.gridn);
    Eigen::SparseMatrix<double> star = getStarMatrix(simdata.gridn, simdata.gridh);
    Eigen::SparseMatrix<double> laplacian = (-1 * simdata.M_inv * vertD.transpose() * star * vertD);

    // plug into solver
    int n = simdata.gridn;
    Eigen::SparseMatrix<double> reg(n * n, n * n);
    reg.setIdentity();

    Eigen::SparseMatrix<double> vertComp = reg - ((simdata.viscosity * simdata.timestep / simdata.density) * laplacian);
    simdata.vertSolver.compute(vertComp);
}

void computeCellLaplacian(SimulationData& simdata) {
    int n = simdata.gridn - 1;
    Eigen::SparseMatrix<double> cellD = getDMatrix(n);
    Eigen::SparseMatrix<double> i(n * n, n * n);
    i.setIdentity();
    Eigen::SparseMatrix<double> laplacian = (-1.0 / (simdata.gridh * simdata.gridh)) * cellD.transpose() * cellD - 0.00005 * i;
    simdata.cellSolver.compute(laplacian);
}

void computeLaplacians(SimulationData& simdata) {
    computeVertLaplacian(simdata);
    computeCellLaplacian(simdata);
}

void handleParamUpdate(SimulationData& simdata) {
    updateMassInv(simdata);
    computeLaplacians(simdata);
}

// Initialize a new simulation

void makeGrid(int gridn, SimulationData& result)
{
    result.gridn = gridn;
    result.gridh = 2.0 / double(gridn - 1);
    result.V.resize(gridn * gridn, 3);
    for (int i = 0; i < gridn; i++)
    {
        for (int j = 0; j < gridn; j++)
        {
            double x = -1.0 + 2.0 * double(j) / double(gridn - 1);
            double y = -1.0 + 2.0 * double(i) / double(gridn - 1);
            result.V(i * gridn + j, 0) = x;
            result.V(i * gridn + j, 1) = y;
            result.V(i * gridn + j, 2) = 0;
        }
    }
    result.F.resize(2 * (gridn - 1) * (gridn - 1), 3);
    for (int i = 0; i < gridn - 1; i++)
    {
        for (int j = 0; j < gridn - 1; j++)
        {
            int idx = 2 * (i * (gridn - 1) + j);
            result.F(idx, 0) = i * gridn + j;
            result.F(idx, 1) = i * gridn + (j + 1);
            result.F(idx, 2) = (i + 1) * gridn + (j + 1);
            result.F(idx + 1, 0) = i * gridn + j;
            result.F(idx + 1, 1) = (i + 1) * gridn + (j + 1);
            result.F(idx + 1, 2) = (i + 1) * gridn + j;
        }
    }

    result.u.resize(gridn * gridn, 2);
    result.u.setZero();

    result.p.resize((gridn - 1) * (gridn - 1));
    result.p.setZero();

    result.externalForce.resize(gridn * gridn, 2);
    result.externalForce.setZero();

    result.markerParticles.resize(result.nparticles, 3);
    result.markerParticles.setRandom();
    result.markerParticles.col(2).setZero();

    // TODO: any extra initialization you want to do

    handleParamUpdate(result);
}

Eigen::Vector2d interpolate(Eigen::Vector2d thisPt, int botLeftIdx, int botRightIdx, int topLeftIdx, int topRightIdx, SimulationData &simdata) {
    Eigen::Vector2d ret;

    // get actual locations of corners
    Eigen::Vector2d botLeft = simdata.V.row(botLeftIdx).segment(0, 2);
    Eigen::Vector2d botRight = simdata.V.row(botRightIdx).segment(0, 2);
    Eigen::Vector2d topLeft = simdata.V.row(topLeftIdx).segment(0, 2);
    Eigen::Vector2d topRight = simdata.V.row(topRightIdx).segment(0, 2);

    // get diffs
    double xDiff1 = topRight(0) - thisPt(0);
    double yDiff1 = topRight(1) - thisPt(1);
    double xDiff2 = thisPt(0) - botLeft(0);
    double yDiff2 = thisPt(1) - botLeft(1);

    // compute interpolation of u
    Eigen::MatrixXd &u = simdata.u;
    ret(0) = (u(botLeftIdx, 0) * xDiff1 * yDiff1 +
              u(botRightIdx, 0) * xDiff2 * yDiff1 +
              u(topLeftIdx, 0) * xDiff1 * yDiff2 +
              u(topRightIdx, 0) * xDiff2 * yDiff2) / 
            ((topRight(0) - botLeft(0)) * (topRight(1) - botLeft(1)));
    ret(1) = (u(botLeftIdx, 1) * xDiff1 * yDiff1 +
              u(botRightIdx, 1) * xDiff2 * yDiff1 +
              u(topLeftIdx, 1) * xDiff1 * yDiff2 +
              u(topRightIdx, 1) * xDiff2 * yDiff2) / 
            ((topRight(0) - botLeft(0)) * (topRight(1) - botLeft(1)));

    return ret;
}

Eigen::MatrixXd semiLagrangianAdvection(SimulationData& simdata) {
    int n = simdata.gridn;
    Eigen::MatrixXd uAdv;
    uAdv.resize(n * n, 2);
    uAdv.setZero();

    // loop through points
    for (int p_i = 0; p_i < n * n; p_i++) {
        // get the new point
        Eigen::Vector2d oldVel = simdata.u.row(p_i);
        Eigen::Vector2d oldPoint = simdata.V.row(p_i).segment(0, 2);
        Eigen::Vector2d newPoint = oldPoint - simdata.timestep * oldVel;

        // find the cell containing this point
        if (newPoint(0) < -1 || newPoint(0) > 1 || 
            newPoint(1) < -1 || newPoint(1) > 1)
            continue;
        if (newPoint(0) >= 1.0 - simdata.gridh / 3.0)
            newPoint(0) -= simdata.gridh / 3.0;
        if (newPoint(1) >= 1.0 - simdata.gridh / 3.0)
            newPoint(1) -= simdata.gridh / 3.0;
        int cellX = (int) std::floor((newPoint(0) + 1.0) / simdata.gridh);
        int cellY = (int) std::floor((newPoint(1) + 1.0) / simdata.gridh);

        // cell vertex indexes
        int botLeftIdx = cellY * simdata.gridn + cellX;
        int botRightIdx = cellY * simdata.gridn + cellX + 1;
        int topLeftIdx = (cellY + 1) * simdata.gridn + cellX;
        int topRightIdx = (cellY + 1) * simdata.gridn + cellX + 1;

        // update with interpolation
        Eigen::Vector2d newV = interpolate(newPoint, botLeftIdx, botRightIdx, topLeftIdx, topRightIdx, simdata);
        uAdv.row(p_i) = newV;
    }
    return uAdv;
}

Eigen::MatrixXd unconstrainedStep(SimulationData& simdata, Eigen::MatrixXd u_adv) {
    Eigen::MatrixXd right = u_adv + ((simdata.timestep / simdata.density) * simdata.externalForce);

    Eigen::MatrixXd v;
    v.resize(simdata.gridn * simdata.gridn, 2);
    v.col(0) = simdata.vertSolver.solve(right.col(0));
    v.col(1) = simdata.vertSolver.solve(right.col(1));
    return v;
}

int toIdx(int i, int j, int n) {
    return i * n + j;
}

Eigen::MatrixXd projectBoundary(SimulationData& simdata, Eigen::MatrixXd u) {
    int n = simdata.gridn;

    for (int i = 0; i < n; i++) {
        u(toIdx(n - 1, i, n), 1) = 0;
        u(toIdx(0, i, n), 1) = 0;
        u(toIdx(i, 0, n), 0) = 0;
        u(toIdx(i, n - 1, n), 0) = 0;
    }

    return u;
}

void computePressure(SimulationData& simdata, Eigen::MatrixXd &u) {
    int n = simdata.gridn - 1;
    Eigen::VectorXd divergence;
    divergence.resize(n * n);
    divergence.setZero();

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            // get 4 corners of this cell
            int botLeftIdx = i * simdata.gridn + j;
            int botRightIdx = i * simdata.gridn + j + 1;
            int topLeftIdx = ((i + 1) * simdata.gridn) + j;
            int topRightIdx = ((i + 1) * simdata.gridn) + j + 1;

            // get edges
            double leftEdge = (u(botLeftIdx, 0) + u(topLeftIdx, 0));
            double rightEdge = (u(botRightIdx, 0) + u(topRightIdx, 0));
            double botEdge = (u(botLeftIdx, 1) + u(botRightIdx, 1));
            double topEdge = (u(topLeftIdx, 1) + u(topRightIdx, 1));

            // update cell's divergence
            divergence(i * n + j) += (rightEdge - leftEdge + topEdge - botEdge) / (2 * simdata.gridh);
        }
    }

    simdata.p = simdata.cellSolver.solve(divergence);
}

void updateVelocity(SimulationData& simdata, Eigen::MatrixXd u) {
    // loop through vertices
    for (int i = 0; i < simdata.gridn; ++i) {
        for (int j = 0; j < simdata.gridn; ++j) {
            int idx = toIdx(i, j, simdata.gridn);

            // get cells next to vertex
            int botLeftIdx = (i - 1) * (simdata.gridn - 1) + j - 1;
            int botRightIdx = (i - 1) * (simdata.gridn - 1) + j;
            int topLeftIdx = i * (simdata.gridn - 1) + j - 1;
            int topRightIdx = i * (simdata.gridn - 1) + j;

            // update u
            if (isCorner(i, j, simdata.gridn))
                continue;
            else if (i == 0)
                u(idx, 0) -= (simdata.p(topRightIdx) - simdata.p(topLeftIdx)) / simdata.gridh / 2;
            else if (i == simdata.gridn - 1)
                u(idx, 0) -= (simdata.p(botRightIdx) - simdata.p(botLeftIdx)) / simdata.gridh / 2;
            else if (j == 0)
                u(idx, 1) -= (simdata.p(topRightIdx) - simdata.p(botRightIdx)) / simdata.gridh / 2;
            else if (j == simdata.gridn - 1)
                u(idx, 1) -= (simdata.p(topLeftIdx) - simdata.p(botLeftIdx)) / simdata.gridh / 2;
            else {
                u(idx, 0) -= (simdata.p(topRightIdx) - simdata.p(topLeftIdx) + simdata.p(botRightIdx) - simdata.p(botLeftIdx)) / simdata.gridh / 2;
                u(idx, 1) -= (simdata.p(topRightIdx) - simdata.p(botRightIdx) + simdata.p(topLeftIdx) - simdata.p(botLeftIdx)) / simdata.gridh / 2;
            }
        }
    }

    simdata.u = u;
}

void advectMarkerParticles(SimulationData& simdata) {
    // loop through particles
    for (int i = 0; i < simdata.nparticles; i++) {
        // get particle pos
        Eigen::Vector2d pt = simdata.markerParticles.row(i).segment(0, 2);

        // find the cell containing this point
        if (pt(0) >= 1.0 - simdata.gridh / 3.0)
            pt(0) -= simdata.gridh / 3.0;
        if (pt(1) >= 1.0 - simdata.gridh / 3.0)
            pt(1) -= simdata.gridh / 3.0;
        int cellX = (int) std::floor((pt(0) + 1.0) / simdata.gridh);
        int cellY = (int) std::floor((pt(1) + 1.0) / simdata.gridh);

        // cell vertex indexes
        int botLeftIdx = cellY * simdata.gridn + cellX;
        int botRightIdx = cellY * simdata.gridn + cellX + 1;
        int topLeftIdx = (cellY + 1) * simdata.gridn + cellX;
        int topRightIdx = (cellY + 1) * simdata.gridn + cellX + 1;

        // interpolate velocity, update pos
        Eigen::Vector2d vel = interpolate(pt, botLeftIdx, botRightIdx, topLeftIdx, topRightIdx, simdata);
        pt += vel * simdata.timestep;
        if (pt(0) < -1 || pt(0) > 1 || 
            pt(1) < -1 || pt(1) > 1)
            pt.setZero();
        simdata.markerParticles.row(i).segment(0, 2) = pt;
    }
}

void simulateOneStep(SimulationData& simdata) {
    // TODO: all substeps of the fluid simulation
    // 1. Use semi-Lagrangian advection to compute the inertial update to u
    // 2. Solve a linear system to compute the unconstrained new velocity field (accounting for intertia, viscosity, and external forces but not pressure)
    // 3. Project the velocity field to ensure the fluid stays inside the box
    // 4. Solve a linear system to compute pressure
    // 5. Apply the pressure gradient to the velocity field

    Eigen::MatrixXd u_adv = semiLagrangianAdvection(simdata);
    Eigen::MatrixXd u = projectBoundary(simdata, unconstrainedStep(simdata, u_adv));
    computePressure(simdata, u);
    updateVelocity(simdata, u);
    
    // 6. Flow marker particles along the velocity field
    advectMarkerParticles(simdata);
}

int main(int argc, char *argv[])
{
    polyscope::init();

    SimulationData simdata;
    simdata.timestep = 0.10;
    simdata.density = 1.0;
    simdata.viscosity = 0.0;
    simdata.mouseStrength = 100.0;
    simdata.nparticles = 1000;

    makeGrid(20, simdata);

    // Set up rendering

    polyscope::view::style = polyscope::NavigateStyle::Planar;
    polyscope::view::projectionMode = polyscope::ProjectionMode::Orthographic;
    
    auto* pmesh = polyscope::registerSurfaceMesh("Mesh", simdata.V, simdata.F);
    pmesh->addVertexVectorQuantity2D("External Force", simdata.externalForce);    
    auto *vfield = pmesh->addVertexVectorQuantity2D("Velocity Field", simdata.u, polyscope::VectorType::AMBIENT);
    vfield->setEnabled(true);
    vfield->setVectorLengthScale(0.1);

    Eigen::VectorXd facecolors;
    makeFaceColors(simdata, facecolors);
    auto *pfield = pmesh->addFaceScalarQuantity("Pressure", facecolors);
    pfield->setEnabled(true);
    pfield->setMapRange({ -1e-2, 1e-2 });

    auto* mparticles = polyscope::registerPointCloud("Marker Particles", simdata.markerParticles);
    mparticles->setEnabled(true);
    mparticles->setPointColor({ 1,1,1 });
    mparticles->setPointRadius(0.003);

    // GUI state

    bool isdragging = false;
    float prevx = 0;
    float prevy = 0;

    polyscope::state::userCallback = [&]()->void
    {
        bool meshdirty = false;
        
        int oldsize = simdata.gridn;
        if (ImGui::InputInt("Grid Size", &oldsize))
        {
            makeGrid(oldsize, simdata);
            meshdirty = true;
        }

        double oldTimeStep = simdata.timestep;
        double oldDensity = simdata.density;
        double oldViscosity = simdata.viscosity;
        double oldMouseStrength = simdata.mouseStrength;

        ImGui::InputDouble("Time step", &simdata.timestep);
        ImGui::InputDouble("Density", &simdata.density);
        ImGui::InputDouble("Viscosity", &simdata.viscosity);
        ImGui::InputDouble("Drag Strength", &simdata.mouseStrength);

        if (oldTimeStep != simdata.timestep || 
            oldDensity != simdata.density || 
            oldViscosity != simdata.viscosity ||
            oldMouseStrength != simdata.mouseStrength) {
            handleParamUpdate(simdata);
        }

        simdata.externalForce.setZero();

        if (ImGui::IsMouseDown(0))
        {
            ImGuiIO& io = ImGui::GetIO();
            float mousex = io.MousePos.x;
            float mousey = io.MousePos.y;

            float mousexndc = 2.0 * io.MousePos.x / polyscope::view::windowWidth - 1.0;
            float mouseyndc = 1.0 - 2.0 * io.MousePos.y / polyscope::view::windowHeight;
            glm::vec4 mousendc(mousexndc, mouseyndc, 0.0, 1.0);
            
            glm::vec4 mouseworld = glm::inverse(pmesh->getTransform()) * glm::inverse(polyscope::view::getCameraViewMatrix()) * glm::inverse(polyscope::view::getCameraPerspectiveMatrix()) * mousendc;
            float worldx = mouseworld[0] / mouseworld[3];
            float worldy = mouseworld[1] / mouseworld[3];
            if (isdragging)
            {
                float deltax = worldx - prevx;
                float deltay = worldy - prevy;
                setExternalForce(simdata, worldx, worldy, deltax, deltay);
            }
            else
            {
                isdragging = true;
                
            }
            prevx = worldx;
            prevy = worldy;
        }
        else
        {
            isdragging = false;
        }

        simulateOneStep(simdata);
        

        // Refresh all rendered geometry

        if (meshdirty)
        {
            pmesh = polyscope::registerSurfaceMesh("Mesh", simdata.V, simdata.F);
        }
        pmesh->addVertexVectorQuantity2D("External Force", simdata.externalForce);
        vfield = pmesh->addVertexVectorQuantity2D("Velocity Field", simdata.u, polyscope::VectorType::AMBIENT);
        makeFaceColors(simdata, facecolors);
        pfield = pmesh->addFaceScalarQuantity("Pressure", facecolors);        
        pfield->setMapRange({ -1e-2, 1e-2 });
        mparticles = polyscope::registerPointCloud("Marker Particles", simdata.markerParticles);
    };
    
    polyscope::show();
}
