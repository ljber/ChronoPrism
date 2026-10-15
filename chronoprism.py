#!/usr/bin/env python3
"""
ChronoPrism Ray Tracing Lab v4 (OpenGL 3.3 Core + NVIDIA Dedicated GPU)
-----------------------------------------------------------------------
Interactive 3D Optical Ray Tracing Sandbox & Physics Lab.
New in v4:
  - Redesigned Right-Side Lab Dock with unified dynamic layout engine (zero text/slider overlap)
  - Clean 2-Column Physics & Laser Slider Grid with generous label-to-track clearance
  - Full Mouse-Wheel Sidebar Scrolling support for smaller window sizes
  - Untruncated button labels & refined visual hierarchy
  - Dedicated Offscreen High-Resolution Photo Renderer (QHD, 4K UHD, 8K UHD) with 4x RGSS,
    live progress bar modal, and automatic OS image launch

Install dependencies:
    pip install moderngl pygame numpy
Run:
    python chronoprism_raylab_v4.py
"""

import os
import sys
import math
import time
import ctypes
import subprocess
from datetime import datetime
import numpy as np

# 1. Force Dedicated NVIDIA GPU Selection on Linux Prime & Windows Hybrid Systems
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["SHIM_MCCOMPAT"] = "0x800000001"

if sys.platform == "win32":
    try:
        ctypes.WinDLL("nvapi64.dll")
    except Exception:
        pass

import pygame
import moderngl

# ==============================================================================
# GLSL 330 CORE SHADERS (Multi-Geometry Ray Tracing + 4x RGSS High-Res Mode)
# ==============================================================================

VERTEX_SHADER = """
#version 330 core
in vec2 in_vert;
out vec2 v_uv;
void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform vec2 uResolution;
uniform vec3 uCamPos;
uniform vec3 uCamTarget;
uniform float uTimeOfDay;
uniform vec3 uSunDir;
uniform vec3 uMoonDir;
uniform vec3 uSunColor;
uniform vec3 uZenithColor;
uniform vec3 uHorizonColor;
uniform float uStarVis;
uniform float uSunIntensity;
uniform float uMoonIntensity;

// Optical Lab Uniforms
uniform int uShapeType;          // 0:Cube, 1:Sphere, 2:Octahedron, 3:Torus, 4:Prism, 5:Diamond Gem
uniform float uIOR;
uniform float uDispersion;
uniform float uBevel;
uniform float uRoughness;
uniform float uAbsorption;
uniform vec3 uGlassTint;
uniform float uCausticsIntensity;
uniform int uMaxBounces;
uniform float uShellThickness;   // 0.0 = Solid, >0.0 = Hollow Shell
uniform float uTwist;            // Spatial twist / wave deformation
uniform float uExposure;
uniform int uFloorPattern;
uniform int uShowTotem;
uniform float uCubeRotY;
uniform float uCubeRotX;

// Laser Optical Bench Uniforms
uniform int uLaserEnabled;
uniform float uLaserAngle;
uniform float uLaserElev;
uniform float uLaserHeight;
uniform float uLaserWavelength;  // 0.0..1.0 = spectral, <0 = white laser

// High-Res Offscreen Anti-Aliasing Samples (1 = Real-Time, 4 = 4x RGSS Supersampling)
uniform int uAASamples;

const float PI = 3.14159265359;

mat2 rot2D(float a) {
    float s = sin(a), c = cos(a);
    return mat2(c, -s, s, c);
}

float hash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

vec3 wavelengthToRGB(float t) {
    return clamp(vec3(
        1.5 - abs(4.0 * t - 1.0),
        1.5 - abs(4.0 * t - 2.0),
        1.5 - abs(4.0 * t - 3.0)
    ), 0.0, 1.0);
}

// --- SDF Primitives ---
float sdRoundBox(vec3 p, vec3 b, float r) {
    vec3 q = abs(p) - (b - vec3(r));
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

float sdOctahedron(vec3 p, float s) {
    p = abs(p);
    float m = p.x + p.y + p.z - s;
    vec3 q;
    if (3.0 * p.x < m) q = p.xyz;
    else if (3.0 * p.y < m) q = p.yzx;
    else if (3.0 * p.z < m) q = p.zxy;
    else return m * 0.57735027;
    float k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    return length(vec3(q.x, q.y - s + k, q.z - k));
}

float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xy) - t.x, p.z);
    return length(q) - t.y;
}

float sdTriPrism(vec3 p, vec2 h, float r) {
    vec3 q = abs(p);
    float d1 = q.z - h.y;
    float d2 = max(q.x * 0.866025 + p.y * 0.5, -p.y) - h.x * 0.5;
    return length(max(vec2(d1, d2), 0.0)) + min(max(d1, d2), 0.0) - r;
}

float sdCutGem(vec3 p, float r) {
    vec3 q = p;
    float d = dot(abs(q), normalize(vec3(1.0, 1.15, 1.0))) - 0.92;
    q.xz *= rot2D(PI * 0.25);
    float d2 = dot(abs(q), normalize(vec3(1.0, 1.15, 1.0))) - 0.92;
    d = max(d, d2);
    d = max(d, p.y - 0.55);
    d = max(d, -p.y - 0.88);
    return d - r * 0.35;
}

float sdShapeOuter(vec3 p) {
    p.xz *= rot2D(uCubeRotY);
    p.yz *= rot2D(uCubeRotX);

    if (abs(uTwist) > 0.001) {
        p.xz *= rot2D(p.y * uTwist * 1.4);
    }

    float d = 0.0;
    if (uShapeType == 0) {
        d = sdRoundBox(p, vec3(0.92), uBevel);
    } else if (uShapeType == 1) {
        d = length(p) - 0.98;
    } else if (uShapeType == 2) {
        d = sdOctahedron(p, 1.32 - uBevel) - uBevel;
    } else if (uShapeType == 3) {
        d = sdTorus(p, vec2(0.68, 0.32));
    } else if (uShapeType == 4) {
        d = sdTriPrism(p, vec2(1.12, 0.88), uBevel);
    } else {
        d = sdCutGem(p, uBevel);
    }

    if (abs(uTwist) > 0.001) {
        d *= 0.82;
    }
    return d;
}

float sdObject(vec3 p) {
    float dOuter = sdShapeOuter(p);
    if (uShellThickness > 0.01) {
        float cavity = -(dOuter + uShellThickness);
        return max(dOuter, cavity);
    }
    return dOuter;
}

vec3 calcNormal(vec3 p) {
    const float eps = 0.0006;
    vec2 e = vec2(1.0, -1.0) * 0.5773 * eps;
    return normalize(
        e.xyy * sdObject(p + e.xyy) +
        e.yyx * sdObject(p + e.yyx) +
        e.yxy * sdObject(p + e.yxy) +
        e.xxx * sdObject(p + e.xxx)
    );
}

vec2 objectAABB(vec3 ro, vec3 rd) {
    vec3 invR = 1.0 / rd;
    vec3 tbot = invR * (vec3(-1.55) - ro);
    vec3 ttop = invR * (vec3(1.55) - ro);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    float t0 = max(max(tmin.x, tmin.y), tmin.z);
    float t1 = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(max(t0, 0.0), t1);
}

vec3 evalSky(vec3 rd) {
    float y = clamp(rd.y, 0.0, 1.0);
    vec3 sky = mix(uHorizonColor, uZenithColor, pow(y, 0.55));

    float sunDot = max(dot(rd, uSunDir), 0.0);
    if (uSunDir.y > -0.18) {
        float horizonFade = smoothstep(-0.18, 0.08, uSunDir.y);
        sky += uSunColor * pow(sunDot, 6.0) * 0.45 * horizonFade;
        sky += uSunColor * pow(sunDot, 64.0) * 0.9 * horizonFade;
        float sunDisc = smoothstep(0.99935, 0.99975, sunDot);
        sky += uSunColor * sunDisc * 9.0 * horizonFade;
    }

    float moonDot = max(dot(rd, uMoonDir), 0.0);
    if (uMoonDir.y > -0.1) {
        float mFade = smoothstep(-0.1, 0.1, uMoonDir.y) * uMoonIntensity;
        sky += vec3(0.45, 0.60, 0.95) * pow(moonDot, 22.0) * 0.35 * mFade;
        float moonDisc = smoothstep(0.99945, 0.99978, moonDot);
        float crater = 0.85 + 0.15 * sin(rd.x * 90.0) * cos(rd.y * 90.0);
        sky += vec3(0.85, 0.92, 1.0) * moonDisc * crater * 4.5 * mFade;
    }

    if (uStarVis > 0.01 && rd.y > 0.02) {
        vec3 sdir = rd;
        sdir.xz *= rot2D(uTimeOfDay * 0.2618);
        vec2 suv = sdir.xz / (sdir.y + 0.15) * 42.0;
        vec2 id = floor(suv);
        vec2 gv = fract(suv) - 0.5;
        float rnd = hash21(id);
        if (rnd > 0.945) {
            vec2 starOffset = (vec2(fract(rnd * 34.0), fract(rnd * 71.0)) - 0.5) * 0.6;
            float d = length(gv - starOffset);
            float star = smoothstep(0.09, 0.0, d) * fract(rnd * 113.0);
            vec3 sCol = mix(vec3(0.75, 0.85, 1.0), vec3(1.0, 0.9, 0.75), fract(rnd * 19.0));
            sky += sCol * star * uStarVis * smoothstep(0.02, 0.28, rd.y) * 1.4;
        }
        float mw = exp(-pow((sdir.x * 0.8 + sdir.z * 0.5) * 2.5, 2.0));
        sky += vec3(0.18, 0.24, 0.42) * mw * uStarVis * 0.22 * smoothstep(0.05, 0.5, rd.y);
    }

    if (rd.y < 0.0) {
        vec3 nadirCol = uZenithColor * 0.18 + uHorizonColor * 0.22;
        sky = mix(uHorizonColor * 0.75, nadirCol, clamp(-rd.y * 2.5, 0.0, 1.0));
    }
    return max(sky, vec3(0.0));
}

void getLaserSetup(out vec3 lOrigin, out vec3 lDir, out vec3 lColor) {
    float ca = cos(uLaserAngle), sa = sin(uLaserAngle);
    float ce = cos(uLaserElev),  se = sin(uLaserElev);
    lOrigin = vec3(-2.65 * ca, uLaserHeight, -2.65 * sa);
    lDir = normalize(vec3(ca * ce, se, sa * ce));
    if (uLaserWavelength < 0.0) {
        lColor = vec3(1.0, 0.98, 0.95);
    } else {
        lColor = wavelengthToRGB(uLaserWavelength);
    }
}

vec2 intersectCylinder(vec3 ro, vec3 rd, vec3 cPos, float r, float hMin, float hMax) {
    vec2 oc = ro.xz - cPos.xz;
    float a = dot(rd.xz, rd.xz);
    float b = 2.0 * dot(oc, rd.xz);
    float c = dot(oc, oc) - r * r;
    float disc = b * b - 4.0 * a * c;
    if (disc < 0.0) return vec2(-1.0);
    float sq = sqrt(disc);
    float t1 = (-b - sq) / (2.0 * a);
    float t2 = (-b + sq) / (2.0 * a);
    float y1 = ro.y + t1 * rd.y;
    if (t1 > 0.001 && y1 >= hMin && y1 <= hMax) return vec2(t1, 1.0);
    float y2 = ro.y + t2 * rd.y;
    if (t2 > 0.001 && y2 >= hMin && y2 <= hMax) return vec2(t2, 1.0);
    return vec2(-1.0);
}

vec3 evalTotemColor(vec3 p, vec3 n) {
    float band = fract(p.y * 2.2 - 0.15);
    float angle = atan(n.z, n.x);
    vec3 baseCol;
    if (band < 0.25) baseCol = vec3(0.95, 0.22, 0.28);
    else if (band < 0.50) baseCol = vec3(1.00, 0.72, 0.15);
    else if (band < 0.75) baseCol = vec3(0.12, 0.78, 0.65);
    else baseCol = vec3(0.25, 0.48, 0.96);

    float stripe = smoothstep(-0.05, 0.05, sin(angle * 6.0 + p.y * 5.0));
    baseCol = mix(baseCol, vec3(0.96, 0.97, 1.0), stripe * 0.55);

    vec3 lDir = uSunIntensity > 0.05 ? uSunDir : uMoonDir;
    vec3 lCol = uSunIntensity > 0.05 ? uSunColor * uSunIntensity : vec3(0.45, 0.6, 0.95) * uMoonIntensity;
    float diff = max(dot(n, lDir), 0.0);
    vec3 amb = mix(uHorizonColor, uZenithColor, 0.5) * 0.55 + vec3(0.08);
    return baseCol * (amb + lCol * diff * 0.9) + baseCol * 0.18;
}

vec3 evalFloorMaterial(vec2 p) {
    float dist = length(p);
    if (uFloorPattern == 0) {
        vec2 grid1 = abs(fract(p * 1.25 - 0.5) - 0.5) / fwidth(p * 1.25);
        float line1 = 1.0 - min(min(grid1.x, grid1.y), 1.0);
        vec2 grid2 = abs(fract(p * 0.25 - 0.5) - 0.5) / fwidth(p * 0.25);
        float line2 = 1.0 - min(min(grid2.x, grid2.y), 1.0);
        vec3 base = vec3(0.14, 0.16, 0.21);
        base = mix(base, vec3(0.26, 0.31, 0.40), line1 * 0.5);
        base = mix(base, vec3(0.42, 0.60, 0.85), line2 * 0.65);
        return base;
    } else if (uFloorPattern == 1) {
        vec2 id = floor(p * 0.8);
        float chk = mod(id.x + id.y, 2.0);
        float vein = sin(p.x * 3.5 + sin(p.y * 4.2) * 1.8) * 0.5 + 0.5;
        vec3 c1 = mix(vec3(0.76, 0.78, 0.82), vec3(0.55, 0.58, 0.65), pow(vein, 4.0));
        vec3 c2 = mix(vec3(0.16, 0.18, 0.23), vec3(0.30, 0.34, 0.42), pow(vein, 5.0));
        return mix(c2, c1, chk);
    } else if (uFloorPattern == 2) {
        float rings = abs(fract(dist * 1.1) - 0.5) / fwidth(dist * 1.1);
        float rLine = 1.0 - min(rings, 1.0);
        float angle = atan(p.y, p.x);
        float spokes = abs(fract(angle * 12.0 / (2.0 * PI)) - 0.5) / max(fwidth(angle * 2.0), 0.001);
        float sLine = 1.0 - min(spokes, 1.0);
        vec3 base = mix(vec3(0.13, 0.15, 0.20), vec3(0.20, 0.23, 0.30), smoothstep(0.0, 4.0, dist));
        base = mix(base, vec3(0.75, 0.64, 0.42), max(rLine * 0.6, sLine * 0.35));
        return base;
    } else if (uFloorPattern == 3) {
        float stripe = fract((p.x + p.y * 0.35) * 0.65);
        vec3 spec = wavelengthToRGB(stripe);
        float mask = smoothstep(0.35, 0.4, abs(fract((p.x + p.y * 0.35) * 3.25) - 0.5));
        return mix(vec3(0.16, 0.18, 0.23), spec * 0.85 + 0.15, 1.0 - mask * 0.6);
    } else {
        vec2 g = abs(fract(p * 2.0 - 0.5) - 0.5) / fwidth(p * 2.0);
        float gl = 1.0 - min(min(g.x, g.y), 1.0);
        return mix(vec3(0.05, 0.06, 0.08), vec3(0.12, 0.16, 0.24), gl * 0.45);
    }
}

vec3 evalFloorAndCaustics(vec3 p) {
    float radialFade = exp(-dot(p.xz, p.xz) * 0.028);
    vec3 matCol = evalFloorMaterial(p.xz);

    bool useSun = uSunDir.y > -0.02;
    vec3 lightDir = useSun ? uSunDir : (uMoonDir.y > 0.0 ? uMoonDir : vec3(0.0, 1.0, 0.0));
    vec3 lightCol = useSun ? uSunColor * uSunIntensity : vec3(0.45, 0.62, 0.95) * uMoonIntensity;
    float lightActive = useSun ? smoothstep(-0.02, 0.08, uSunDir.y) : smoothstep(0.0, 0.12, uMoonDir.y);

    float NdotL = max(lightDir.y, 0.0);
    vec3 skyAmb = mix(uHorizonColor, uZenithColor, 0.45) * 0.42 + vec3(0.04, 0.05, 0.07);

    float shadowTrans = 1.0;
    vec3 causticGlow = vec3(0.0);

    if (lightActive > 0.001 && lightDir.y > 0.03) {
        float tHit = (0.0 - p.y) / lightDir.y;
        vec3 projCenter = p + lightDir * tHit;

        float shapeD = sdObject(projCenter);
        float soft = 0.08 + (1.0 - lightDir.y) * 0.22;
        float inShadow = 1.0 - smoothstep(-soft, soft, shapeD);

        shadowTrans = mix(1.0, 0.26, inShadow);
        vec3 glassFilter = mix(vec3(1.0), uGlassTint, inShadow * 0.65);

        float dispSpread = (0.06 + uDispersion * 2.5) * (1.25 - lightDir.y * 0.45) * (uIOR - 1.0);
        vec2 caustDir = normalize(lightDir.xz + vec2(0.0001));

        for (int i = 0; i < 5; i++) {
            float fi = float(i) / 4.0;
            float shift = (fi - 0.5) * dispSpread;
            vec3 sampleP = projCenter;
            sampleP.xz += caustDir * shift * 1.4;

            float dSample = sdObject(sampleP);
            float focusRim = exp(-abs(dSample + 0.16) * 8.0);
            float crossWave = pow(max(0.0, cos(sampleP.x * 5.0 + float(uShapeType)) * cos(sampleP.z * 5.0)), 2.0);
            float coreBeam = exp(-max(dSample + 0.35, 0.0) * 5.2) * (0.65 + 0.65 * crossWave);

            vec3 specCol = wavelengthToRGB(fi);
            causticGlow += specCol * (focusRim * 1.2 + coreBeam * 0.85) * 0.42;
        }

        causticGlow *= uCausticsIntensity * lightActive * (uSunIntensity * 1.25 + uMoonIntensity * 0.65);
        causticGlow *= uGlassTint;
        matCol *= glassFilter;
    }

    if (uLaserEnabled == 1) {
        vec3 lOrig, lDir, lCol;
        getLaserSetup(lOrig, lDir, lCol);
        vec2 exitCenter = lDir.xz * (1.55 + (uIOR - 1.0) * 0.65);
        for (int k = 0; k < 5; k++) {
            float fk = float(k) / 4.0;
            float spread = (fk - 0.5) * uDispersion * 12.0 * (uIOR - 1.0);
            vec2 perp = vec2(-lDir.z, lDir.x);
            vec2 spotPos = exitCenter + lDir.xz * (fk - 0.5) * 0.7 + perp * spread;
            float dSpot = length(p.xz - spotPos);
            vec3 cSpot = (uLaserWavelength < 0.0) ? wavelengthToRGB(fk) : lCol;
            causticGlow += cSpot * exp(-dSpot * dSpot * 14.0) * 2.2 * uCausticsIntensity;
        }
    }

    float ao = smoothstep(0.0, 1.55, length(p.xz));
    ao = mix(0.38, 1.0, ao);

    vec3 directLight = lightCol * NdotL * shadowTrans;
    vec3 color = matCol * (skyAmb * ao + directLight) + causticGlow * (0.48 + 0.52 * matCol);

    vec3 horizonFog = mix(uHorizonColor * 0.85, uZenithColor * 0.35, 0.3);
    return mix(horizonFog, color, radialFade);
}

vec3 sampleEnv(vec3 ro, vec3 rd) {
    float tBest = 1e9;
    int hitType = 0;

    if (uShowTotem == 1) {
        vec3 cPos = vec3(0.0, 0.0, -3.1);
        vec2 cyl = intersectCylinder(ro, rd, cPos, 0.42, -0.95, 1.45);
        if (cyl.x > 0.0 && cyl.x < tBest) {
            tBest = cyl.x;
            hitType = 2;
        }
    }

    if (rd.y < -0.0005) {
        float tf = (-0.95 - ro.y) / rd.y;
        if (tf > 0.001 && tf < 45.0 && tf < tBest) {
            tBest = tf;
            hitType = 1;
        }
    }

    if (hitType == 2) {
        vec3 p = ro + rd * tBest;
        vec3 n = normalize(vec3(p.x, 0.0, p.z + 3.1));
        return evalTotemColor(p, n);
    } else if (hitType == 1) {
        vec3 p = ro + rd * tBest;
        return evalFloorAndCaustics(p);
    }
    return evalSky(rd);
}

float fresnelSchlick(float cosTheta, float ior) {
    float r0 = (1.0 - ior) / (1.0 + ior);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float distToRay(vec3 p, vec3 ro, vec3 rd, out float sProj) {
    vec3 v = p - ro;
    sProj = dot(v, rd);
    if (sProj < 0.0) return 1e5;
    return length(v - rd * sProj);
}

vec3 traceInternalRay(vec3 entryPos, vec3 entryNormal, vec3 viewDir, float channelIOR, float wlFactor) {
    vec3 rd = refract(viewDir, entryNormal, 1.0 / channelIOR);
    if (length(rd) < 0.001) {
        return sampleEnv(entryPos + entryNormal * 0.003, reflect(viewDir, entryNormal));
    }

    vec3 ro = entryPos - entryNormal * 0.004;
    vec3 accumCol = vec3(0.0);
    vec3 throughput = vec3(1.0);
    float totalDist = 0.0;

    vec3 lOrig, lDir, lCol;
    vec3 lInternalOrig = vec3(0.0);
    vec3 lInternalDir = vec3(1.0, 0.0, 0.0);
    if (uLaserEnabled == 1) {
        getLaserSetup(lOrig, lDir, lCol);
        lInternalOrig = lOrig + lDir * 1.75;
        vec3 approxN = -lDir;
        lInternalDir = normalize(refract(lDir, approxN, 1.0 / channelIOR) + vec3(0.0, (wlFactor - 0.5) * uDispersion * 2.5, 0.0));
    }

    for (int bounce = 0; bounce < 10; bounce++) {
        if (bounce >= uMaxBounces) break;

        float t = 0.012;
        for (int s = 0; s < 52; s++) {
            float d = -sdObject(ro + rd * t);
            if (d < 0.0008 || t > 5.0) break;
            t += max(d, 0.002);
        }

        if (uLaserEnabled == 1) {
            for (int ls = 1; ls <= 4; ls++) {
                vec3 sp = ro + rd * (t * float(ls) * 0.22);
                float sProj;
                float dBeam = distToRay(sp, lInternalOrig, lInternalDir, sProj);
                if (sProj > -1.2 && sProj < 2.5) {
                    vec3 beamCol = (uLaserWavelength < 0.0) ? wavelengthToRGB(wlFactor) : lCol;
                    accumCol += throughput * beamCol * exp(-dBeam * dBeam * 140.0) * 0.85;
                }
            }
        }

        totalDist += t;
        vec3 hitPos = ro + rd * t;
        vec3 outNormal = calcNormal(hitPos);
        vec3 intNormal = -outNormal;

        vec3 absorb = exp(-(vec3(1.02) - uGlassTint) * totalDist * uAbsorption);

        float cosI = clamp(dot(-rd, intNormal), 0.0, 1.0);
        float F = fresnelSchlick(cosI, channelIOR);

        vec3 refractOut = refract(rd, intNormal, channelIOR);
        vec3 reflectIn = reflect(rd, intNormal);

        if (uRoughness > 0.001) {
            vec3 roughPerturb = sin(hitPos * 38.0) * uRoughness * 0.35;
            refractOut = normalize(refractOut + roughPerturb);
        }

        if (length(refractOut) < 0.001) {
            ro = hitPos + intNormal * 0.004;
            rd = reflectIn;
            throughput *= absorb * 0.97;
        } else {
            vec3 exitEnv = sampleEnv(hitPos + outNormal * 0.004, refractOut);
            float specSun = pow(max(dot(refractOut, uSunDir), 0.0), 120.0) * uSunIntensity;
            exitEnv += uSunColor * specSun * 1.8;

            accumCol += throughput * (1.0 - F) * absorb * exitEnv;
            throughput *= F * absorb;
            ro = hitPos + intNormal * 0.004;
            rd = reflectIn;
            if (max(throughput.r, max(throughput.g, throughput.b)) < 0.025) break;
        }
    }

    if (max(throughput.r, max(throughput.g, throughput.b)) >= 0.025) {
        accumCol += throughput * sampleEnv(ro, rd);
    }
    return accumCol;
}

vec3 evalExteriorLaserVolumetric(vec3 ro, vec3 rd, float maxT) {
    if (uLaserEnabled == 0) return vec3(0.0);
    vec3 lOrig, lDir, lCol;
    getLaserSetup(lOrig, lDir, lCol);

    vec3 w0 = ro - lOrig;
    float a = dot(rd, rd);
    float b = dot(rd, lDir);
    float c = dot(lDir, lDir);
    float d = dot(rd, w0);
    float e = dot(lDir, w0);
    float denom = a * c - b * b;
    if (abs(denom) < 0.0001) return vec3(0.0);

    float sc = (b * e - c * d) / denom;
    float tc = (a * e - b * d) / denom;

    vec3 glow = vec3(0.0);
    if (sc > 0.05 && sc < maxT && tc > 0.0 && tc < 6.0) {
        vec3 pCam = ro + rd * sc;
        vec3 pLas = lOrig + lDir * tc;
        float dist = length(pCam - pLas);
        float core = exp(-dist * dist * 1600.0) * 2.8;
        float halo = exp(-dist * dist * 75.0) * 0.45;
        glow += lCol * (core + halo);
    }

    float tEmitter = dot(lOrig - ro, rd);
    if (tEmitter > 0.0 && tEmitter < maxT) {
        float dEm = length((ro + rd * tEmitter) - lOrig);
        glow += lCol * exp(-dEm * dEm * 450.0) * 3.5;
    }
    return glow;
}

vec3 acesToneMap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 renderRaySample(vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * uResolution.xy) / uResolution.y;

    vec3 ro = uCamPos;
    vec3 ta = uCamTarget;
    vec3 ww = normalize(ta - ro);
    vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
    vec3 vv = normalize(cross(uu, ww));
    vec3 rd = normalize(uv.x * uu + uv.y * vv + 1.65 * ww);

    vec2 boxHit = objectAABB(ro, rd);
    bool hitObj = false;
    float tObj = 0.0;

    if (boxHit.x < boxHit.y && boxHit.y > 0.0) {
        float t = max(boxHit.x, 0.01);
        for (int i = 0; i < 76; i++) {
            float d = sdObject(ro + rd * t);
            if (d < 0.0007) {
                hitObj = true;
                tObj = t;
                break;
            }
            t += d;
            if (t > boxHit.y) break;
        }
    }

    float tFloor = 1e9;
    if (rd.y < -0.0005) {
        float tf = (-0.95 - ro.y) / rd.y;
        if (tf > 0.001) tFloor = tf;
    }

    vec3 col = vec3(0.0);
    float maxRayDist = min(tFloor, 20.0);

    if (hitObj && tObj < tFloor) {
        maxRayDist = tObj;
        vec3 p = ro + rd * tObj;
        vec3 n = calcNormal(p);

        if (uRoughness > 0.001) {
            vec3 nPerturb = sin(p * 42.0) * cos(p.zxy * 42.0) * uRoughness * 0.25;
            n = normalize(n + nPerturb);
        }

        float cosTheta = clamp(dot(-rd, n), 0.0, 1.0);
        float F = fresnelSchlick(cosTheta, uIOR);

        vec3 reflDir = reflect(rd, n);
        vec3 reflCol = sampleEnv(p + n * 0.004, reflDir);

        float sunSpec = pow(max(dot(reflDir, uSunDir), 0.0), mix(220.0, 24.0, uRoughness)) * uSunIntensity;
        float moonSpec = pow(max(dot(reflDir, uMoonDir), 0.0), mix(180.0, 20.0, uRoughness)) * uMoonIntensity;
        reflCol += uSunColor * sunSpec * 3.5 + vec3(0.65, 0.8, 1.0) * moonSpec * 1.4;

        float dSpread = uDispersion * 1.35;
        float iorR = uIOR - dSpread;
        float iorY = uIOR - dSpread * 0.5;
        float iorG = uIOR;
        float iorC = uIOR + dSpread * 0.5;
        float iorB = uIOR + dSpread;

        vec3 cR = traceInternalRay(p, n, rd, iorR, 0.1);
        vec3 cG = traceInternalRay(p, n, rd, iorG, 0.5);
        vec3 cB = traceInternalRay(p, n, rd, iorB, 0.9);

        vec3 refrCol;
        if (uDispersion > 0.003) {
            vec3 cY = traceInternalRay(p, n, rd, iorY, 0.3);
            vec3 cC = traceInternalRay(p, n, rd, iorC, 0.7);
            refrCol.r = (cR.r * 0.65 + cY.r * 0.35);
            refrCol.g = (cY.g * 0.25 + cG.g * 0.50 + cC.g * 0.25);
            refrCol.b = (cC.b * 0.35 + cB.b * 0.65);
        } else {
            refrCol = vec3(cR.r, cG.g, cB.b);
        }

        vec3 edgeHighlight = (uSunColor * uSunIntensity * 0.35 + uHorizonColor * 0.25) * pow(1.0 - cosTheta, 3.0);
        col = mix(refrCol, reflCol, F) + edgeHighlight;
    } else {
        col = sampleEnv(ro, rd);
    }

    col += evalExteriorLaserVolumetric(ro, rd, maxRayDist);

    float vig = 1.0 - 0.20 * dot(uv, uv);
    col *= vig;
    col = acesToneMap(col * uExposure);
    return col;
}

void main() {
    vec3 totalCol = vec3(0.0);
    if (uAASamples <= 1) {
        totalCol = renderRaySample(gl_FragCoord.xy);
    } else {
        const vec2 rgss[4] = vec2[4](
            vec2( 0.125,  0.375),
            vec2(-0.375,  0.125),
            vec2( 0.375, -0.125),
            vec2(-0.125, -0.375)
        );
        for (int s = 0; s < 4; s++) {
            totalCol += renderRaySample(gl_FragCoord.xy + rgss[s]);
        }
        totalCol *= 0.25;
    }
    totalCol = pow(totalCol, vec3(1.0 / 2.2));
    fragColor = vec4(totalCol, 1.0);
}
"""

# ==============================================================================
# SOLAR & ATMOSPHERIC CYCLE + MATERIAL / GEOMETRY / RESOLUTION PRESETS
# ==============================================================================

SKY_KEYFRAMES = [
    (0.0,  (0.015, 0.025, 0.065), (0.04, 0.065, 0.14),  (1.0, 0.5, 0.2),  "Midnight",            "1800K", 0.95),
    (4.5,  (0.025, 0.04, 0.11),   (0.09, 0.11, 0.24),   (1.0, 0.4, 0.2),  "Astronomical Dawn",   "2200K", 0.70),
    (5.5,  (0.08, 0.13, 0.32),    (0.55, 0.28, 0.32),   (1.0, 0.42, 0.18),"Civil Twilight",      "2700K", 0.25),
    (6.25, (0.16, 0.32, 0.65),    (0.98, 0.48, 0.22),   (1.0, 0.56, 0.24),"Sunrise · Golden Hr", "3100K", 0.02),
    (8.0,  (0.14, 0.44, 0.85),    (0.68, 0.82, 0.96),   (1.0, 0.88, 0.72),"Morning",             "4600K", 0.00),
    (12.0, (0.09, 0.38, 0.88),    (0.62, 0.84, 1.00),   (1.0, 0.97, 0.92),"Solar Noon",          "5800K", 0.00),
    (16.0, (0.14, 0.42, 0.84),    (0.75, 0.82, 0.92),   (1.0, 0.86, 0.68),"Afternoon",           "4800K", 0.00),
    (18.0, (0.20, 0.26, 0.58),    (0.99, 0.42, 0.18),   (1.0, 0.46, 0.16),"Sunset · Golden Hr",  "2900K", 0.05),
    (19.0, (0.07, 0.10, 0.28),    (0.45, 0.22, 0.38),   (0.9, 0.35, 0.25),"Dusk · Blue Hour",    "2400K", 0.45),
    (20.5, (0.02, 0.03, 0.09),    (0.06, 0.09, 0.18),   (0.8, 0.4, 0.2),  "Nightfall",           "1900K", 0.88),
    (24.0, (0.015, 0.025, 0.065), (0.04, 0.065, 0.14),  (1.0, 0.5, 0.2),  "Midnight",            "1800K", 0.95),
]

SHAPES = [
    "Beveled Cube",
    "Optical Sphere",
    "Octahedron",
    "Torus Ring",
    "Equilateral Prism",
    "Brilliant Gem",
]

MATERIALS = [
    ("Pure Water",   1.33, 0.009, (0.90, 0.97, 1.00)),
    ("Glacial Ice",  1.31, 0.008, (0.82, 0.94, 1.00)),
    ("Fused Quartz", 1.46, 0.014, (0.97, 0.98, 1.00)),
    ("Crown Glass",  1.52, 0.022, (0.93, 0.98, 0.96)),
    ("Flint Glass",  1.65, 0.042, (0.96, 0.97, 1.00)),
    ("Sapphire",     1.77, 0.030, (0.72, 0.86, 1.00)),
    ("Emerald",      1.58, 0.026, (0.68, 0.98, 0.78)),
    ("Diamond",      2.42, 0.068, (0.98, 0.99, 1.00)),
]

FLOOR_NAMES = ["Optical Grid", "Marble Tile", "Sundial Rings", "Chromatic Bars", "Obsidian Bench"]

# Offscreen High-Res Presets: (Label, Width, Height, AA Samples, Tile Grid X, Tile Grid Y)
RENDER_PRESETS = [
    ("QHD 2560x1440 (4x RGSS)", 2560, 1440, 4, 4, 4),
    ("4K UHD 3840x2160 (4x RGSS)", 3840, 2160, 4, 6, 6),
    ("8K UHD 7680x4320 (4x RGSS)", 7680, 4320, 4, 8, 8),
]

def lerp_tuple(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(len(a)))

def compute_atmosphere(hour):
    h = hour % 24.0
    k0, k1 = SKY_KEYFRAMES[0], SKY_KEYFRAMES[-1]
    for i in range(len(SKY_KEYFRAMES) - 1):
        if SKY_KEYFRAMES[i][0] <= h <= SKY_KEYFRAMES[i + 1][0]:
            k0, k1 = SKY_KEYFRAMES[i], SKY_KEYFRAMES[i + 1]
            break
    span = max(k1[0] - k0[0], 0.0001)
    t = (h - k0[0]) / span
    st = t * t * (3.0 - 2.0 * t)

    zenith = lerp_tuple(k0[1], k1[1], st)
    horizon = lerp_tuple(k0[2], k1[2], st)
    sun_col = lerp_tuple(k0[3], k1[3], st)
    star_vis = k0[6] + (k1[6] - k0[6]) * st
    phase_info = k0 if t < 0.5 else k1

    hour_angle = ((h - 6.0) / 24.0) * math.pi * 2.0
    elev = math.sin(hour_angle) * 1.18
    azim = -math.cos(hour_angle) * 1.25
    sun_dir = np.array([math.cos(elev) * math.sin(azim), math.sin(elev), math.cos(elev) * math.cos(azim)], dtype=np.float32)
    sun_dir /= np.linalg.norm(sun_dir)

    moon_angle = hour_angle + math.pi
    m_elev = math.sin(moon_angle) * 0.95
    m_azim = -math.cos(moon_angle) * 1.15
    moon_dir = np.array([math.cos(m_elev) * math.sin(m_azim), math.sin(m_elev), math.cos(m_elev) * math.cos(m_azim)], dtype=np.float32)
    moon_dir /= np.linalg.norm(moon_dir)

    sun_int = float(np.clip((sun_dir[1] + 0.08) / 0.33, 0.0, 1.0)) * 1.35
    moon_int = float(np.clip((moon_dir[1] + 0.05) / 0.30, 0.0, 1.0)) * (1.0 - min(sun_int, 1.0)) * 0.85

    return {
        "zenith": zenith,
        "horizon": horizon,
        "sun_col": sun_col,
        "sun_dir": tuple(sun_dir),
        "moon_dir": tuple(moon_dir),
        "star_vis": star_vis,
        "sun_int": sun_int,
        "moon_int": moon_int,
        "phase": phase_info[4],
        "kelvin": phase_info[5],
        "elev_deg": round(math.degrees(math.asin(float(np.clip(sun_dir[1], -1.0, 1.0))))),
    }

def format_time(h):
    total_mins = int(round((h % 24.0) * 60)) % 1440
    hrs = total_mins // 60
    mins = total_mins % 60
    ampm = "AM" if hrs < 12 else "PM"
    h12 = hrs % 12
    if h12 == 0:
        h12 = 12
    return f"{hrs:02d}:{mins:02d} ({h12}:{mins:02d} {ampm})"

def open_file_in_os(filepath):
    """Automatically launches the rendered image in the user's default OS viewer."""
    abs_path = os.path.abspath(filepath)
    try:
        if sys.platform == "win32":
            os.startfile(abs_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", abs_path])
        else:
            subprocess.Popen(["xdg-open", abs_path])
        return True
    except Exception as e:
        print(f"[ChronoPrism] Could not auto-open file ({e}). Saved at: {abs_path}")
        return False

# ==============================================================================
# INTERACTIVE RAY TRACING LAB APPLICATION
# ==============================================================================

def main():
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

    width, height = 1480, 860
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)

    ctx = moderngl.create_context()
    gpu_vendor = ctx.info.get("GL_VENDOR", "Unknown Vendor")
    gpu_renderer = ctx.info.get("GL_RENDERER", "Unknown GPU")
    pygame.display.set_caption(f"ChronoPrism Ray Tracing Lab v4 | GPU: {gpu_renderer}")

    print("=" * 78)
    print(f"[ChronoPrism Ray Lab v4] Active GPU: {gpu_renderer} ({gpu_vendor})")
    print("=" * 78)

    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

    quad_vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
    vbo = ctx.buffer(quad_vertices.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, "in_vert")

    hud_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    hud_tex = ctx.texture((width, height), 4)
    hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    HUD_VERT = """
    #version 330 core
    in vec2 in_vert;
    out vec2 v_uv;
    void main() {
        v_uv = vec2(in_vert.x * 0.5 + 0.5, 1.0 - (in_vert.y * 0.5 + 0.5));
        gl_Position = vec4(in_vert, 0.0, 1.0);
    }
    """
    HUD_FRAG = """
    #version 330 core
    in vec2 v_uv;
    out vec4 fragColor;
    uniform sampler2D uHudTex;
    void main() {
        fragColor = texture(uHudTex, v_uv);
    }
    """
    hud_prog = ctx.program(vertex_shader=HUD_VERT, fragment_shader=HUD_FRAG)
    hud_vao = ctx.simple_vertex_array(hud_prog, vbo, "in_vert")

    font_title = pygame.font.SysFont("Segoe UI, Arial", 20, bold=True)
    font_sec = pygame.font.SysFont("Segoe UI, Arial", 12, bold=True)
    font_mono = pygame.font.SysFont("Consolas, Courier New", 12, bold=True)
    font_sm = pygame.font.SysFont("Segoe UI, Arial", 12)

    state = {
        "time_of_day": 9.85,
        "auto_play": False,
        "play_speed": 1.0,
        "shape_type": 0,
        "mat_idx": 3,              # Crown Glass
        "ior": MATERIALS[3][1],
        "dispersion": MATERIALS[3][2],
        "glass_tint": list(MATERIALS[3][3]),
        "bevel": 0.065,
        "roughness": 0.0,
        "absorption": 0.55,
        "caustics": 1.25,
        "max_bounces": 5,
        "shell_thickness": 0.0,
        "twist": 0.0,
        "exposure": 1.08,
        "floor_pattern": 0,
        "show_totem": 1,
        "auto_rot": False,
        "cube_rot_y": 0.62,
        "cube_rot_x": 0.0,
        "laser_enabled": 0,
        "laser_angle": 0.25,
        "laser_elev": -0.08,
        "laser_height": 0.15,
        "laser_wavelength": -1.0,
        "render_preset_idx": 1,    # Default: 4K UHD 3840x2160 (4x RGSS)
        "show_ui": True,
        "sidebar_scroll": 0,
        "sidebar_max_scroll": 0,
        "status_msg": "Press [P] or click 'RENDER HIGH-RES PHOTO' for an offscreen 4K render with auto-open!",
        "status_timer": 5.0,
    }

    render_job = {
        "active": False,
        "fbo": None,
        "tex": None,
        "w": 3840,
        "h": 2160,
        "aa": 4,
        "label": "",
        "tiles": [],
        "current_tile": 0,
        "total_tiles": 0,
        "start_time": 0.0,
    }

    cam_yaw = 0.68
    cam_pitch = 0.34
    cam_dist = 4.25
    dragging_cam = False
    dragging_obj = False
    dragging_time = False
    active_slider_key = None

    # Compact 2-Column Friendly Labels so every slider has generous breathing room
    SLIDER_DEFS = [
        ("ior",             "Refractive IOR",       1.05, 2.60, "{:.2f}"),
        ("dispersion",      "Dispersion",           0.00, 0.09, "{:.3f}"),
        ("max_bounces",     "Max Bounces",          1.0,  10.0, "{:.0f}"),
        ("bevel",           "Edge Bevel",           0.005,0.28, "{:.3f}"),
        ("roughness",       "Roughness",            0.00, 0.50, "{:.2f}"),
        ("absorption",      "Beer Absorb",          0.00, 2.50, "{:.2f}"),
        ("shell_thickness", "Hollow Shell",         0.00, 0.45, "{:.2f}"),
        ("twist",           "Crystal Twist",        -1.5, 1.50, "{:.2f}"),
        ("caustics",        "Caustic Power",        0.00, 2.50, "{:.2f}"),
        ("exposure",        "Exposure",             0.40, 2.20, "{:.2f}"),
        ("laser_angle",     "Laser Azimuth",       -3.14, 3.14, "{:.2f}"),
        ("laser_height",    "Laser Height",        -0.75, 0.85, "{:.2f}"),
        ("laser_wavelength","Laser Color (<0 Wht)",-0.15, 1.00, "{:.2f}"),
    ]

    clock = pygame.time.Clock()
    running = True

    def set_status(msg, duration=4.0):
        state["status_msg"] = msg
        state["status_timer"] = duration

    def start_highres_render():
        if render_job["active"]:
            return
        label, rw, rh, aa, gx, gy = RENDER_PRESETS[state["render_preset_idx"]]
        try:
            off_tex = ctx.texture((rw, rh), 3)
            off_fbo = ctx.framebuffer(color_attachments=[off_tex])
        except Exception as e:
            set_status(f"GPU VRAM allocation failed for {rw}x{rh}: {e}")
            return

        tiles = []
        tw = math.ceil(rw / gx)
        th = math.ceil(rh / gy)
        for ty_idx in range(gy):
            for tx_idx in range(gx):
                x0 = tx_idx * tw
                y0 = ty_idx * th
                w0 = min(tw, rw - x0)
                h0 = min(th, rh - y0)
                if w0 > 0 and h0 > 0:
                    tiles.append((x0, y0, w0, h0))

        render_job["active"] = True
        render_job["fbo"] = off_fbo
        render_job["tex"] = off_tex
        render_job["w"] = rw
        render_job["h"] = rh
        render_job["aa"] = aa
        render_job["label"] = label
        render_job["tiles"] = tiles
        render_job["current_tile"] = 0
        render_job["total_tiles"] = len(tiles)
        render_job["start_time"] = time.perf_counter()

    def cancel_highres_render(msg="High-res render cancelled."):
        if render_job["fbo"] is not None:
            render_job["fbo"].release()
        if render_job["tex"] is not None:
            render_job["tex"].release()
        render_job["fbo"] = None
        render_job["tex"] = None
        render_job["active"] = False
        ctx.screen.use()
        ctx.viewport = (0, 0, width, height)
        set_status(msg)

    def finish_highres_render():
        rw, rh = render_job["w"], render_job["h"]
        raw = render_job["fbo"].read(components=3)
        img = pygame.image.fromstring(raw, (rw, rh), "RGB", True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"chronoprism_{rw}x{rh}_{timestamp}.png"
        pygame.image.save(img, fname)
        elapsed = time.perf_counter() - render_job["start_time"]

        render_job["fbo"].release()
        render_job["tex"].release()
        render_job["fbo"] = None
        render_job["tex"] = None
        render_job["active"] = False

        ctx.screen.use()
        ctx.viewport = (0, 0, width, height)

        opened = open_file_in_os(fname)
        abs_path = os.path.abspath(fname)
        if opened:
            set_status(f"Rendered {rw}x{rh} in {elapsed:.2f}s & opened: {fname}", 6.0)
        else:
            set_status(f"Rendered {rw}x{rh} in {elapsed:.2f}s -> Saved to {abs_path}", 6.0)
        print(f"[ChronoPrism] High-Res Render Complete ({rw}x{rh} in {elapsed:.2f}s): {abs_path}")

    while running:
        dt = clock.tick(60) / 1000.0
        if state["status_timer"] > 0:
            state["status_timer"] = max(0.0, state["status_timer"] - dt)

        panel_w = 384 if state["show_ui"] else 0
        viewport_w = width - panel_w
        time_slider_rect = pygame.Rect(40, height - 54, max(200, viewport_w - 80), 20)

        # ----------------------------------------------------------------------
        # Unified Dynamic Sidebar Layout Engine (Zero Hardcoded Overlaps!)
        # ----------------------------------------------------------------------
        ui_headers = []
        ui_buttons = []
        ui_sliders = {}
        header_banner_h = 62

        if state["show_ui"]:
            px = width - panel_w + 18
            pw = panel_w - 36
            bw = (pw - 10) // 2
            bh = 26
            row_gap = 6
            scroll_y = state["sidebar_scroll"]
            y_cursor = header_banner_h + 10 - scroll_y

            # Section 1: Dielectric Geometry
            ui_headers.append((px, y_cursor, "1. DIELECTRIC GEOMETRY [G]", (220, 232, 252)))
            y_cursor += 22
            for i, sname in enumerate(SHAPES):
                bx = px + (i % 2) * (bw + 10)
                by = y_cursor + (i // 2) * (bh + row_gap)
                ui_buttons.append((pygame.Rect(bx, by, bw, bh), "shape", i, sname))
            y_cursor += 3 * (bh + row_gap) + 12

            # Section 2: Optical Material Presets
            ui_headers.append((px, y_cursor, "2. OPTICAL MATERIAL PRESETS [M]", (220, 232, 252)))
            y_cursor += 22
            for i, mdata in enumerate(MATERIALS):
                bx = px + (i % 2) * (bw + 10)
                by = y_cursor + (i // 2) * (bh + row_gap)
                ui_buttons.append((pygame.Rect(bx, by, bw, bh), "mat", i, mdata[0]))
            y_cursor += 4 * (bh + row_gap) + 12

            # Section 3: Bench Tools & Environment
            ui_headers.append((px, y_cursor, "3. BENCH TOOLS & ENVIRONMENT", (220, 232, 252)))
            y_cursor += 22
            toggles = [
                ("laser", f"Laser Bench: {'ON' if state['laser_enabled'] else 'OFF'}"),
                ("hollow", f"Hollow Bubble: {'ON' if state['shell_thickness'] > 0.01 else 'OFF'}"),
                ("totem", f"Refract Totem: {'ON' if state['show_totem'] else 'OFF'}"),
                ("floor", f"Floor: {FLOOR_NAMES[state['floor_pattern']]}"),
            ]
            for i, (tkey, tlabel) in enumerate(toggles):
                bx = px + (i % 2) * (bw + 10)
                by = y_cursor + (i // 2) * (bh + row_gap)
                ui_buttons.append((pygame.Rect(bx, by, bw, bh), "toggle", tkey, tlabel))
            y_cursor += 2 * (bh + row_gap) + 12

            # Section 4: Offscreen High-Res Studio
            ui_headers.append((px, y_cursor, "4. OFFSCREEN HIGH-RES STUDIO [P]", (255, 215, 120)))
            y_cursor += 22
            preset_label = RENDER_PRESETS[state["render_preset_idx"]][0]
            ui_buttons.append((pygame.Rect(px, y_cursor, pw, 25), "res_preset", 0, f"Target: {preset_label}"))
            y_cursor += 31
            ui_buttons.append((pygame.Rect(px, y_cursor, pw, 30), "trigger_render", 0, "RENDER HIGH-RES PHOTO [P]"))
            y_cursor += 42

            # Section 5: Physics & Laser Parameters (Clean 2-Column Grid + Full-Width Last Row)
            ui_headers.append((px, y_cursor, "5. PHYSICS & LASER PARAMETERS", (220, 232, 252)))
            y_cursor += 24

            col_w = (pw - 14) // 2
            slider_row_h = 40
            num_sliders = len(SLIDER_DEFS)

            for idx, (skey, slabel, smin, smax, sfmt) in enumerate(SLIDER_DEFS):
                row = idx // 2
                col = idx % 2
                # Make the last odd slider span full width neatly
                if idx == num_sliders - 1 and num_sliders % 2 == 1:
                    sx = px
                    sw = pw
                else:
                    sx = px + col * (col_w + 14)
                    sw = col_w
                sy_label = y_cursor + row * slider_row_h
                # Place the track 20px below the label top so the thumb never touches the text
                srect = pygame.Rect(sx, sy_label + 20, sw, 8)
                ui_sliders[skey] = (srect, sy_label, smin, smax, slabel, sfmt)

            total_rows = (num_sliders + 1) // 2
            content_bottom = (y_cursor + scroll_y) + total_rows * slider_row_h + 24
            state["sidebar_max_scroll"] = max(0, content_bottom - height)
            state["sidebar_scroll"] = int(np.clip(state["sidebar_scroll"], 0, state["sidebar_max_scroll"]))

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                if render_job["active"]:
                    cancel_highres_render("Window resized; render cancelled.")
                width, height = max(event.w, 960), max(event.h, 640)
                pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
                ctx.viewport = (0, 0, width, height)
                hud_surf = pygame.Surface((width, height), pygame.SRCALPHA)
                hud_tex.release()
                hud_tex = ctx.texture((width, height), 4)

            elif render_job["active"]:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    cancel_highres_render("High-res render cancelled by user.")
                continue

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if event.button == 1:
                    if time_slider_rect.inflate(0, 14).collidepoint(mx, my):
                        dragging_time = True
                        rel_x = np.clip((mx - time_slider_rect.x) / time_slider_rect.w, 0.0, 1.0)
                        state["time_of_day"] = float(rel_x * 24.0)
                    elif state["show_ui"] and mx >= width - panel_w:
                        if my > header_banner_h:
                            clicked_btn = False
                            for brect, btype, bval, _ in ui_buttons:
                                if brect.collidepoint(mx, my):
                                    clicked_btn = True
                                    if btype == "shape":
                                        state["shape_type"] = bval
                                        set_status(f"Switched geometry to {SHAPES[bval]}")
                                    elif btype == "mat":
                                        state["mat_idx"] = bval
                                        _, state["ior"], state["dispersion"], tint = MATERIALS[bval]
                                        state["glass_tint"] = list(tint)
                                        set_status(f"Loaded material preset: {MATERIALS[bval][0]}")
                                    elif btype == "toggle":
                                        if bval == "laser":
                                            state["laser_enabled"] = 1 - state["laser_enabled"]
                                        elif bval == "hollow":
                                            state["shell_thickness"] = 0.0 if state["shell_thickness"] > 0.01 else 0.14
                                        elif bval == "totem":
                                            state["show_totem"] = 1 - state["show_totem"]
                                        elif bval == "floor":
                                            state["floor_pattern"] = (state["floor_pattern"] + 1) % len(FLOOR_NAMES)
                                    elif btype == "res_preset":
                                        state["render_preset_idx"] = (state["render_preset_idx"] + 1) % len(RENDER_PRESETS)
                                        set_status(f"Offscreen Render Target: {RENDER_PRESETS[state['render_preset_idx']][0]}")
                                    elif btype == "trigger_render":
                                        start_highres_render()
                                    break
                            if not clicked_btn:
                                for skey, (srect, _, smin, smax, _, _) in ui_sliders.items():
                                    if srect.inflate(6, 14).collidepoint(mx, my):
                                        active_slider_key = skey
                                        rel = np.clip((mx - srect.x) / srect.w, 0.0, 1.0)
                                        val = smin + rel * (smax - smin)
                                        state[skey] = int(round(val)) if skey == "max_bounces" else float(val)
                                        break
                    else:
                        dragging_cam = True
                elif event.button == 3:
                    if mx < width - panel_w:
                        dragging_obj = True
                elif event.button == 4:  # Scroll Up
                    if state["show_ui"] and mx >= width - panel_w:
                        state["sidebar_scroll"] = max(0, state["sidebar_scroll"] - 28)
                    else:
                        cam_dist = max(1.8, cam_dist - 0.25)
                elif event.button == 5:  # Scroll Down
                    if state["show_ui"] and mx >= width - panel_w:
                        state["sidebar_scroll"] = min(state["sidebar_max_scroll"], state["sidebar_scroll"] + 28)
                    else:
                        cam_dist = min(9.5, cam_dist + 0.25)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_time = False
                    dragging_cam = False
                    active_slider_key = None
                elif event.button == 3:
                    dragging_obj = False

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                dx, dy = event.rel
                if dragging_time:
                    rel_x = np.clip((mx - time_slider_rect.x) / time_slider_rect.w, 0.0, 1.0)
                    state["time_of_day"] = float(rel_x * 24.0)
                elif active_slider_key and active_slider_key in ui_sliders:
                    srect, _, smin, smax, _, _ = ui_sliders[active_slider_key]
                    rel = np.clip((mx - srect.x) / srect.w, 0.0, 1.0)
                    val = smin + rel * (smax - smin)
                    state[active_slider_key] = int(round(val)) if active_slider_key == "max_bounces" else float(val)
                elif dragging_cam:
                    cam_yaw -= dx * 0.007
                    cam_pitch = float(np.clip(cam_pitch + dy * 0.006, -0.15, 1.42))
                elif dragging_obj:
                    state["cube_rot_y"] += dx * 0.01
                    state["cube_rot_x"] = float(np.clip(state["cube_rot_x"] + dy * 0.01, -1.5, 1.5))

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    state["auto_play"] = not state["auto_play"]
                elif event.key == pygame.K_TAB or event.key == pygame.K_h:
                    state["show_ui"] = not state["show_ui"]
                elif event.key == pygame.K_g:
                    state["shape_type"] = (state["shape_type"] + 1) % len(SHAPES)
                    set_status(f"Geometry: {SHAPES[state['shape_type']]}")
                elif event.key == pygame.K_m:
                    state["mat_idx"] = (state["mat_idx"] + 1) % len(MATERIALS)
                    _, state["ior"], state["dispersion"], tint = MATERIALS[state["mat_idx"]]
                    state["glass_tint"] = list(tint)
                    set_status(f"Material: {MATERIALS[state['mat_idx']][0]}")
                elif event.key == pygame.K_l:
                    state["laser_enabled"] = 1 - state["laser_enabled"]
                    set_status(f"Laser Optical Bench: {'ON' if state['laser_enabled'] else 'OFF'}")
                elif event.key == pygame.K_b:
                    state["shell_thickness"] = 0.0 if state["shell_thickness"] > 0.01 else 0.14
                    set_status(f"Hollow Shell Mode: {'ON' if state['shell_thickness'] > 0.01 else 'OFF'}")
                elif event.key == pygame.K_f:
                    state["floor_pattern"] = (state["floor_pattern"] + 1) % len(FLOOR_NAMES)
                elif event.key == pygame.K_t:
                    state["show_totem"] = 1 - state["show_totem"]
                elif event.key == pygame.K_r:
                    state["auto_rot"] = not state["auto_rot"]
                elif event.key == pygame.K_p:
                    start_highres_render()
                elif event.key == pygame.K_BACKSPACE:
                    state["cube_rot_x"] = 0.0
                    state["twist"] = 0.0
                    state["roughness"] = 0.0
                    state["shell_thickness"] = 0.0
                    set_status("Reset crystal orientation & surface deformations.")
                elif event.key == pygame.K_LEFT:
                    state["time_of_day"] = (state["time_of_day"] - 0.25) % 24.0
                elif event.key == pygame.K_RIGHT:
                    state["time_of_day"] = (state["time_of_day"] + 0.25) % 24.0
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                    presets = [0.0, 6.25, 9.5, 12.0, 18.0, 21.5]
                    state["time_of_day"] = presets[event.key - pygame.K_1]

        if not render_job["active"]:
            if state["auto_play"]:
                state["time_of_day"] = (state["time_of_day"] + dt * 0.85 * state["play_speed"]) % 24.0
            if state["auto_rot"]:
                state["cube_rot_y"] += dt * 0.40

        atmos = compute_atmosphere(state["time_of_day"])
        cx = cam_dist * math.cos(cam_pitch) * math.sin(cam_yaw)
        cy = cam_dist * math.sin(cam_pitch) + 0.05
        cz = cam_dist * math.cos(cam_pitch) * math.cos(cam_yaw)

        # Update Shared Scene Uniforms
        prog["uCamPos"].value = (cx, cy, cz)
        prog["uCamTarget"].value = (0.0, -0.05, 0.0)
        prog["uTimeOfDay"].value = float(state["time_of_day"])
        prog["uSunDir"].value = atmos["sun_dir"]
        prog["uMoonDir"].value = atmos["moon_dir"]
        prog["uSunColor"].value = atmos["sun_col"]
        prog["uZenithColor"].value = atmos["zenith"]
        prog["uHorizonColor"].value = atmos["horizon"]
        prog["uStarVis"].value = float(atmos["star_vis"])
        prog["uSunIntensity"].value = float(atmos["sun_int"])
        prog["uMoonIntensity"].value = float(atmos["moon_int"])

        prog["uShapeType"].value = int(state["shape_type"])
        prog["uIOR"].value = float(state["ior"])
        prog["uDispersion"].value = float(state["dispersion"])
        prog["uBevel"].value = float(state["bevel"])
        prog["uRoughness"].value = float(state["roughness"])
        prog["uAbsorption"].value = float(state["absorption"])
        prog["uGlassTint"].value = tuple(state["glass_tint"])
        prog["uCausticsIntensity"].value = float(state["caustics"])
        prog["uMaxBounces"].value = int(state["max_bounces"])
        prog["uShellThickness"].value = float(state["shell_thickness"])
        prog["uTwist"].value = float(state["twist"])
        prog["uExposure"].value = float(state["exposure"])
        prog["uFloorPattern"].value = int(state["floor_pattern"])
        prog["uShowTotem"].value = int(state["show_totem"])
        prog["uCubeRotY"].value = float(state["cube_rot_y"])
        prog["uCubeRotX"].value = float(state["cube_rot_x"])

        prog["uLaserEnabled"].value = int(state["laser_enabled"])
        prog["uLaserAngle"].value = float(state["laser_angle"])
        prog["uLaserElev"].value = float(state["laser_elev"])
        prog["uLaserHeight"].value = float(state["laser_height"])
        prog["uLaserWavelength"].value = float(state["laser_wavelength"])

        # ----------------------------------------------------------------------
        # A. Process Offscreen High-Res Tile Step (if active)
        # ----------------------------------------------------------------------
        if render_job["active"]:
            fbo = render_job["fbo"]
            rw, rh = render_job["w"], render_job["h"]
            fbo.use()
            fbo.viewport = (0, 0, rw, rh)
            prog["uResolution"].value = (float(rw), float(rh))
            prog["uAASamples"].value = int(render_job["aa"])
            ctx.disable(moderngl.BLEND)

            tx, ty, tw, th = render_job["tiles"][render_job["current_tile"]]
            fbo.scissor = (tx, ty, tw, th)
            vao.render(moderngl.TRIANGLE_STRIP)
            fbo.scissor = None
            ctx.finish()

            render_job["current_tile"] += 1

        # ----------------------------------------------------------------------
        # B. Render Live Interactive Window Viewport
        # ----------------------------------------------------------------------
        ctx.screen.use()
        ctx.viewport = (0, 0, width, height)
        prog["uResolution"].value = (float(width), float(height))
        prog["uAASamples"].value = 1
        ctx.disable(moderngl.BLEND)
        vao.render(moderngl.TRIANGLE_STRIP)

        # ----------------------------------------------------------------------
        # C. Draw Native Lab Overlay & High-Res Progress Bar Modal
        # ----------------------------------------------------------------------
        hud_surf.fill((0, 0, 0, 0))

        # Top Telemetry Bar
        pygame.draw.rect(hud_surf, (8, 12, 22, 195), (14, 12, viewport_w - 28, 44), border_radius=10)
        t_surf = font_title.render(f"{format_time(state['time_of_day'])}  ·  {atmos['phase']}", True, (240, 246, 255))
        hud_surf.blit(t_surf, (28, 20))

        fps_val = int(clock.get_fps())
        tele_str = f"GPU: {gpu_renderer[:24]} | {fps_val} FPS | Sun: {atmos['elev_deg']}° ({atmos['kelvin']})"
        tele_surf = font_mono.render(tele_str, True, (130, 200, 255))
        hud_surf.blit(tele_surf, (viewport_w - tele_surf.get_width() - 28, 27))

        if state["status_timer"] > 0:
            s_surf = font_sec.render(state["status_msg"], True, (255, 230, 140))
            hud_surf.blit(s_surf, (28, 62))

        # Bottom 24-Hour Scrubber Bar
        pygame.draw.rect(hud_surf, (8, 12, 22, 215), (14, height - 92, viewport_w - 28, 78), border_radius=12)
        grad_steps = 60
        step_w = time_slider_rect.w / grad_steps
        for i in range(grad_steps):
            a_s = compute_atmosphere((i / grad_steps) * 24.0)
            c = tuple(int(np.clip(v * 255, 0, 255)) for v in a_s["horizon"])
            r_seg = pygame.Rect(int(time_slider_rect.x + i * step_w), time_slider_rect.y, int(math.ceil(step_w)), time_slider_rect.h)
            pygame.draw.rect(hud_surf, c, r_seg)

        pygame.draw.rect(hud_surf, (190, 215, 255, 180), time_slider_rect, width=2, border_radius=5)
        thumb_x = int(time_slider_rect.x + (state["time_of_day"] / 24.0) * time_slider_rect.w)
        pygame.draw.circle(hud_surf, (255, 255, 255), (thumb_x, time_slider_rect.centery), 11)
        pygame.draw.circle(hud_surf, (30, 120, 255), (thumb_x, time_slider_rect.centery), 6)

        hint_str = (
            "[L-Drag] Orbit Cam  |  [R-Drag] Tilt Crystal  |  [Space] Timelapse  |  "
            "[G] Shape  |  [M] Material  |  [L] Laser  |  [P] Offscreen 4K Render  |  [Tab] Hide UI"
        )
        hint_surf = font_sm.render(hint_str, True, (185, 200, 225))
        hud_surf.blit(hint_surf, (40, height - 28))

        # Right-Side Optical Ray Tracing Lab Dock
        if state["show_ui"]:
            dock_rect = pygame.Rect(width - panel_w, 0, panel_w, height)
            pygame.draw.rect(hud_surf, (10, 14, 26, 236), dock_rect)
            pygame.draw.line(hud_surf, (55, 80, 130, 210), (width - panel_w, 0), (width - panel_w, height), 2)

            px = width - panel_w + 18

            # Clip scrollable controls to region below the pinned dock header
            content_clip = pygame.Rect(width - panel_w, header_banner_h, panel_w, height - header_banner_h)
            hud_surf.set_clip(content_clip)

            # 1. Draw Dynamic Section Headers
            for hx, hy, htext, hcol in ui_headers:
                hud_surf.blit(font_sec.render(htext, True, hcol), (hx, hy))

            # 2. Draw Buttons
            for brect, btype, bval, blabel in ui_buttons:
                is_active = False
                if btype == "shape" and state["shape_type"] == bval:
                    is_active = True
                elif btype == "mat" and state["mat_idx"] == bval:
                    is_active = True
                elif btype == "toggle":
                    if (bval == "laser" and state["laser_enabled"]) or \
                       (bval == "hollow" and state["shell_thickness"] > 0.01) or \
                       (bval == "totem" and state["show_totem"]):
                        is_active = True

                if btype == "trigger_render":
                    bg_col = (215, 115, 25, 245)
                    border_col = (255, 210, 130, 255)
                elif btype == "res_preset":
                    bg_col = (28, 44, 72, 230)
                    border_col = (95, 150, 225, 200)
                else:
                    bg_col = (38, 98, 210, 235) if is_active else (24, 32, 52, 220)
                    border_col = (120, 185, 255, 255) if is_active else (60, 75, 110, 180)

                pygame.draw.rect(hud_surf, bg_col, brect, border_radius=5)
                pygame.draw.rect(hud_surf, border_col, brect, width=1, border_radius=5)

                txt_col = (255, 255, 255) if (is_active or btype == "trigger_render") else (190, 208, 235)
                btxt = font_sec.render(blabel, True, txt_col) if btype == "trigger_render" else font_sm.render(blabel, True, txt_col)
                hud_surf.blit(btxt, (brect.centerx - btxt.get_width() // 2, brect.centery - btxt.get_height() // 2))

            # 3. Draw Sliders (with clear separation between label row and track row)
            for skey, (srect, sy_label, smin, smax, slabel, sfmt) in ui_sliders.items():
                val = state[skey]
                lbl_surf = font_sm.render(slabel, True, (198, 214, 238))
                val_surf = font_mono.render(sfmt.format(val), True, (125, 205, 255))
                hud_surf.blit(lbl_surf, (srect.x, sy_label))
                hud_surf.blit(val_surf, (srect.right - val_surf.get_width(), sy_label + 1))

                pygame.draw.rect(hud_surf, (26, 36, 58, 245), srect, border_radius=4)
                rel = float(np.clip((val - smin) / (smax - smin), 0.0, 1.0))
                fill_w = int(srect.w * rel)
                if fill_w > 0:
                    fill_rect = pygame.Rect(srect.x, srect.y, fill_w, srect.h)
                    pygame.draw.rect(hud_surf, (55, 140, 250, 255), fill_rect, border_radius=4)

                thumb_cx = int(np.clip(srect.x + fill_w, srect.x + 4, srect.right - 4))
                pygame.draw.circle(hud_surf, (245, 250, 255), (thumb_cx, srect.centery), 5)
                pygame.draw.circle(hud_surf, (35, 115, 235), (thumb_cx, srect.centery), 2)

            # Scrollbar indicator if window height is smaller than content
            if state["sidebar_max_scroll"] > 0:
                sb_h = max(30, int((height - header_banner_h) * (height - header_banner_h) / (height + state["sidebar_max_scroll"])))
                sb_y = header_banner_h + int((state["sidebar_scroll"] / state["sidebar_max_scroll"]) * (height - header_banner_h - sb_h))
                pygame.draw.rect(hud_surf, (95, 145, 220, 160), (width - 6, sb_y, 4, sb_h), border_radius=2)

            hud_surf.set_clip(None)

            # Pinned Dock Title Header
            pygame.draw.rect(hud_surf, (12, 17, 32, 248), (width - panel_w + 2, 0, panel_w - 2, header_banner_h))
            pygame.draw.line(hud_surf, (45, 68, 110, 200), (width - panel_w, header_banner_h), (width, header_banner_h), 1)
            hdr = font_title.render("RAY TRACING LAB", True, (120, 195, 255))
            hud_surf.blit(hdr, (px, 10))
            sub = font_sm.render("Real-Time Dielectric & Spectral Sandbox", True, (155, 175, 205))
            hud_surf.blit(sub, (px, 36))

        # High-Res Rendering Progress Bar Modal Overlay
        if render_job["active"]:
            pygame.draw.rect(hud_surf, (4, 8, 16, 165), (0, 0, width, height))

            mw, mh = 520, 175
            mx = (viewport_w - mw) // 2
            my = (height - mh) // 2
            modal_rect = pygame.Rect(mx, my, mw, mh)
            pygame.draw.rect(hud_surf, (12, 18, 34, 245), modal_rect, border_radius=14)
            pygame.draw.rect(hud_surf, (90, 165, 255, 255), modal_rect, width=2, border_radius=14)

            pct = render_job["current_tile"] / max(1, render_job["total_tiles"])
            elapsed = time.perf_counter() - render_job["start_time"]

            m_title = font_title.render("RENDERING HIGH-RES OPTICAL PHOTO...", True, (240, 248, 255))
            hud_surf.blit(m_title, (mx + 26, my + 20))

            m_sub = font_mono.render(
                f"Target: {render_job['w']}x{render_job['h']} ({render_job['aa']}x RGSS Supersampling)",
                True, (135, 205, 255)
            )
            hud_surf.blit(m_sub, (mx + 26, my + 54))

            bar_rect = pygame.Rect(mx + 26, my + 88, mw - 52, 26)
            pygame.draw.rect(hud_surf, (26, 36, 58, 255), bar_rect, border_radius=8)

            fill_w = int(bar_rect.w * pct)
            if fill_w > 0:
                fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, fill_w, bar_rect.h)
                pygame.draw.rect(hud_surf, (45, 145, 255, 255), fill_rect, border_radius=8)

            pygame.draw.rect(hud_surf, (150, 205, 255, 220), bar_rect, width=1, border_radius=8)

            pct_txt = font_mono.render(f"{pct * 100:5.1f}%", True, (255, 255, 255))
            hud_surf.blit(pct_txt, (bar_rect.centerx - pct_txt.get_width() // 2, bar_rect.centery - pct_txt.get_height() // 2))

            foot_str = f"Tile {render_job['current_tile']} / {render_job['total_tiles']}   ·   Elapsed: {elapsed:.1f}s   ·   [ESC] Cancel"
            foot_txt = font_sm.render(foot_str, True, (185, 200, 225))
            hud_surf.blit(foot_txt, (mx + 26, my + 128))

        hud_data = pygame.image.tostring(hud_surf, "RGBA", False)
        hud_tex.write(hud_data)

        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        hud_tex.use(0)
        hud_vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()

        if render_job["active"] and render_job["current_tile"] >= render_job["total_tiles"]:
            finish_highres_render()

    pygame.quit()

if __name__ == "__main__":
    main()
