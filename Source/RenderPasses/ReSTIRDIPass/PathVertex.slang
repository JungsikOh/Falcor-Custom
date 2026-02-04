#include "Utils/Math/MathConstants.slangh"

import Utils.Geometry.GeometryHelpers;

/** Path vertex representing a surface point for light sampling.
    Minimal structure containing only geometric information needed for sampling.
*/
struct PathVertex
{
    float3 pos;        ///< Vertex position in world space.
    float3 faceNormal; ///< Geometry normal at the vertex (zero if not on a surface).
    bool frontFacing;  ///< True if path vertex is on the front-facing side (if on a surface).

    /** Initialize a path vertex.
        \param[in] pos Position in world space.
        \param[in] faceNormal Face normal (default: zero).
        \param[in] frontFacing Front-facing flag (default: true).
    */
    __init(float3 pos, float3 faceNormal = float3(0.f), bool frontFacing = true)
    {
        this.pos = pos;
        this.faceNormal = faceNormal;
        this.frontFacing = frontFacing;
    }

    /** Get position with offset applied in direction of the geometry normal to avoid self-intersection.
        \param[in] rayDir Ray direction.
        \return Returns the offseted position.
    */
    float3 getRayOrigin(float3 rayDir)
    {
        return computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0.f ? faceNormal : -faceNormal);
    }

    /** Returns the oriented face normal.
        \return Face normal flipped to the same side as the view vector.
    */
    float3 getOrientedFaceNormal()
    {
        return frontFacing ? faceNormal : -faceNormal;
    }
};
