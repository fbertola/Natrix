#pragma kernel AddVelocity
#pragma kernel InitBoundaries
#pragma kernel AdvectVelocity
#pragma kernel Divergence
#pragma kernel ClearBuffer
#pragma kernel Poisson
#pragma kernel SubstractGradient
#pragma kernel CalcVorticity
#pragma kernel ApplyVorticity
#pragma kernel Viscosity
#pragma kernel AddCircleObstacle
#pragma kernel AddTriangleObstacle

#define THREAD_COUNT 8

const uint2 _Size;
RWStructuredBuffer<float2> _VelocityIn;
RWStructuredBuffer<float2> _VelocityOut;
RWStructuredBuffer<float2> _Obstacles;
RWStructuredBuffer<float> _Divergence;
RWStructuredBuffer<float> _PressureIn;
RWStructuredBuffer<float> _PressureOut;
RWStructuredBuffer<float> _Vorticity;

float _ElapsedTime;
float _Speed;
float _Radius;
float2 _Position;
float2 _Value;
float _VorticityScale;

// For Obstacles
int _Static;

uint4 GetNeighbours(int2 pos, int2 size)
{
	uint4 result;
	const int maxX = size.x - 1;
	const int maxY = size.y - 1;

	result.x = pos.y*_Size.x + clamp(pos.x - 1, 0, maxX);
	result.y = pos.y*_Size.x + clamp(pos.x + 1, 0, maxX);
	result.z = clamp(pos.y - 1, 0, maxY)*size.x + pos.x;
	result.w = clamp(pos.y + 1, 0, maxY)*size.x + pos.x;
	return result;
}

////////////////////////////////////////////////////////////////////////////
// UTIL: Clear Buffer
RWStructuredBuffer<float2> _Buffer;
[numthreads(THREAD_COUNT, THREAD_COUNT, 1)]
void ClearBuffer(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= _Size.x || id.y >= _Size.y) { return; }
	uint pos = id.y*_Size.x + id.x;
	_Buffer[pos].x = 0.0;
}

////////////////////////////////////////////////////////////////////////////
[numthreads(THREAD_COUNT, THREAD_COUNT, 1)]
void AddVelocity(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= _Size.x || id.y >= _Size.y) { return; }

	const uint pos = id.y*_Size.x + id.x;
	const float2 splat_pos = _Position*_Size;

	const float2 val = _VelocityIn[pos];
	float2 result = val;

	float len = distance(splat_pos, (float2) id);
	if (len <= _Radius)
	{
		result = val + _Value*(_Radius - len) / _Radius;
	}

	_VelocityOut[pos] = clamp(result, float2(-1.0, -1.0), float2(1.0, 1.0));
}

////////////////////////////////////////////////////////////////////////////
[numthreads(THREAD_COUNT, THREAD_COUNT, 1)]
void InitBoundaries(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= _Size.x || id.y >= _Size.y) { return; }
	uint pos = id.y*_Size.x + id.x;

	if (id.x == 0)
	{
		_VelocityIn[pos] = float2(0.0,0.0);
	}
	else if (id.x == _Size.x - 1)
	{
		_VelocityIn[pos] = float2(0.0, 0.0);
	}
	else if (id.y == 0)
	{
		_VelocityIn[pos] = float2(0.0, 0.0);
	}
	else if (id.y == _Size.y - 1)
	{
		_VelocityIn[pos] = float2(0.0, 0.0);
	}
}

////////////////////////////////////////////////////////////////////////////
float _Dissipation;
[numthreads(THREAD_COUNT, THREAD_COUNT, 1)]
void AdvectVelocity(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= _Size.x || id.y >= _Size.y) { return; }
	const uint pos = id.y*_Size.x + id.x;

	const float2 obstacle = _Obstacles[pos];
	if (obstacle.x > 0.0 || obstacle.y > 0.0)
	{
		_VelocityOut[pos] = float2(0, 0);
	}
	else
	{
		const float2 vel = _VelocityIn[pos];
		const float2 final_pos = float2(id.x - vel.x*_ElapsedTime*_Speed, id.y - vel.y*_ElapsedTime*_Speed);

		const int2 zero = int2(0, 0);
		const int2 SizeBounds = int2(_Size.x - 1, _Size.y - 1);
		const int2 top_right = clamp(ceil(final_pos), zero, SizeBounds);
		const int2 bottom_left = clamp(floor(final_pos), zero, SizeBounds);

		const float2 delta = final_pos - bottom_left;

		const float2 lt = _VelocityIn[top_right.y*_Size.x + bottom_left.x];
		const float2 rt = _VelocityIn[top_right.y*_Size.x + top_right.x];

		const float2 lb = _VelocityIn[bottom_left.y*_Size.x + bottom_left.x];
		const float2 rb = _VelocityIn[bottom_left.y*_Size.x + top_right.x];

		const float2 h1 = lerp(lt, rt, delta.x);
		const float2 h2 = lerp(lb, rb, delta.x);

		_VelocityOut[pos] = clamp(lerp(h2, h1, delta.y)*_Dissipation, float2(-1.0, -1.0), float2(1.0, 1.0));
	}
}

////////////////////////////////////////////////////////////////////////////
[numthreads(THREAD_COUNT, THREAD_COUNT, 1)]
void Divergence(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= _Size.x || id.y >= _Size.y) { return; }
	const uint pos = id.y*_Size.x + id.x;

	const uint4 n = GetNeighbours(id.xy, _Size);

	float x1 = _VelocityIn[n.x].x;
	float x2 = _VelocityIn[n.y].x;
	float y1 = _VelocityIn[n.z].y;
	float y2 = _VelocityIn[n.w].y;

	const float2 obsL = _Obstacles[n.x];
	const float2 obsR = _Obstacles[n.y];
	const float2 obsB = _Obstacles[n.z];
	const float2 obsT = _Obstacles[n.w];

	if (obsL.x > 0.0 || obsL.y > 0.0) x1 = 0.0;
	if (obsR.x > 0.0 || obsR.y > 0.0) x2 = 0.0;
	if (obsB.x > 0.0 || obsB.y > 0.0) y1 = 0.0;
	if (obsT.x > 0.0 || obsT.y > 0.0) y2 = 0.0;

	_Divergence[pos] = 0.5f * ((x2 - x1) + (y2 - y1));
}

////////////////////////////////////////////////////////////////////////////
[numthreads(THREAD_COUNT, THREAD_COUNT, 1)]
void Poisson(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= _Size.x || id.y >= _Size.y) { return; }
	const uint pos = id.y*_Size.x + id.x;

	//const float alpha = -1.0f;
	const float rbeta = 0.25;

	const uint4 n = GetNeighbours(id.xy, _Size);

	float p = _PressureIn[pos];


	const float2 obsL = _Obstacles[n.x];
	const float2 obsR = _Obstacles[n.y];
	const float2 obsB = _Obstacles[n.z];
	const float2 obsT = _Obstacles[n.w];

	const float x1 = (obsL.x > 0.0 || obsL.y > 0.0) ? p : _PressureIn[n.x];
	const float x2 = (obsR.x > 0.0 || obsR.y > 0.0) ? p : _PressureIn[n.y];
	const float y1 = (obsB.x > 0.0 || obsB.y > 0.0) ? p : _PressureIn[n.z];
	const float y2 = (obsT.x > 0.0 || obsT.y > 0.0) ? p : _PressureIn[n.w];

	const float b = _Divergence[pos];

	_PressureOut[pos] = (x1 + x2 + y1 + y2 - b) * rbeta;
}

////////////////////////////////////////////////////////////////////////////
[numthreads(THREAD_COUNT, THREAD_COUNT, 1)]
void SubstractGradient(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= _Size.x || id.y >= _Size.y) { return; }

	const uint4 n = GetNeighbours(id.xy, _Size);

	const uint pos = id.y*_Size.x + id.x;
	float x1 = _PressureIn[n.x];
	float x2 = _PressureIn[n.y];
	float y1 = _PressureIn[n.z];
	float y2 = _PressureIn[n.w];
	float p = _PressureIn[pos];

	const float2 obsL = _Obstacles[n.x];
	const float2 obsR = _Obstacles[n.y];
	const float2 obsB = _Obstacles[n.z];
	const float2 obsT = _Obstacles[n.w];

	if (obsL.x > 0.0 || obsL.y > 0.0) x1 = p;
	if (obsR.x > 0.0 || obsR.y > 0.0) x2 = p;
	if (obsB.x > 0.0 || obsB.y > 0.0) y1 = p;
	if (obsT.x > 0.0 || obsT.y > 0.0) y2 = p;

	float2 velocity = _VelocityIn[pos];
	velocity.x -= 0.5f * (x2 - x1);
	velocity.y -= 0.5f * (y2 - y1);
	_VelocityOut[pos] = velocity;
}

////////////////////////////////////////////////////////////////////////////
[numthreads(THREAD_COUNT, THREAD_COUNT, 1)]
void CalcVorticity(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= _Size.x || id.y >= _Size.y) { return; }
	const uint pos = id.y*_Size.x + id.x;
	const uint4 n = GetNeighbours(id.xy, _Size);

	const float2 vL = _VelocityIn[n.x];
	const float2 vR = _VelocityIn[n.y];
	const float2 vB = _VelocityIn[n.z];
	const float2 vT = _VelocityIn[n.w];

	_Vorticity[pos] = 0.5 * ((vR.y - vL.y) - (vT.x - vB.x));
}

////////////////////////////////////////////////////////////////////////////
[numthreads(THREAD_COUNT, THREAD_COUNT, 1)]
void ApplyVorticity(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= _Size.x || id.y >= _Size.y) { return; }
	uint pos = id.y*_Size.x + id.x;

	uint4 n = GetNeighbours(id.xy, _Size);

	float vL = _Vorticity[n.x];
	float vR = _Vorticity[n.y];
	float vB = _Vorticity[n.z];
	float vT = _Vorticity[n.w];
	float vC = _Vorticity[pos];

	float2 force = 0.5 * float2(abs(vT) - abs(vB), abs(vR) - abs(vL));

	float EPSILON = 2.4414e-4;
	float magSqr = max(EPSILON, dot(force, force));
	force = force * rsqrt(magSqr);

	force *= _VorticityScale * vC * float2(1, -1);
	float2 final_force = force * _ElapsedTime;

	_VelocityOut[pos] = _VelocityIn[pos] + float2(final_force.x, final_force.y);
}

////////////////////////////////////////////////////////////////////////////
float _Alpha;
float _rBeta;
[numthreads(THREAD_COUNT, THREAD_COUNT, 1)]
void Viscosity(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= _Size.x || id.y >= _Size.y) { return; }
	const uint pos = id.y*_Size.x + id.x;
	const uint4 n = GetNeighbours(id.xy, _Size);

	const float2 x1 = _VelocityIn[n.x];
	const float2 x2 = _VelocityIn[n.y];
	const float2 y1 = _VelocityIn[n.z];
	const float2 y2 = _VelocityIn[n.w];
	const float2 b = _VelocityIn[pos];

	_VelocityOut[pos] = (x1 + x2 + y1 + y2 + b * _Alpha) * _rBeta;
}


////////////////////////////////////////////////////////////////////////////
[numthreads(THREAD_COUNT, THREAD_COUNT, 1)]
void AddCircleObstacle(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= _Size.x || id.y >= _Size.y) { return; }

	const uint pos = id.y*_Size.x + id.x;
	const float2 splat_pos = _Position*_Size;

	if (distance(splat_pos, (float2) id) <= _Radius)
	{
		if (_Static > 0)
		{
			_Obstacles[pos].y = 1.0;
		}
		else
		{
			_Obstacles[pos].x = 1.0;
		}
	}
}


////////////////////////////////////////////////////////////////////////////
float2 _P1;
float2 _P2;
float2 _P3;
float Sign(float2 p1, float2 p2, float2 p3)
{
	return ((p1.x - p3.x) * (p2.y - p3.y)) - ((p2.x - p3.x) * (p1.y - p3.y));
}

bool IsPointInTriangle(float2 pt, float2 v1, float2 v2, float2 v3)
{
	const bool b1 = Sign(pt, v1, v2) < 0.0f;
	const bool b2 = Sign(pt, v2, v3) < 0.0f;
	const bool b3 = Sign(pt, v3, v1) < 0.0f;
	return (b1 == b2) && (b2 == b3);
}

[numthreads(THREAD_COUNT, THREAD_COUNT, 1)]
void AddTriangleObstacle(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= _Size.x || id.y >= _Size.y) { return; }
	const float2 pt = float2(id.x, id.y) / float2(_Size);
	if (IsPointInTriangle(pt, _P1, _P2, _P3))
	{
		const uint pos = id.y*_Size.x + id.x;
		if (_Static > 0)
		{
			_Obstacles[pos].y = 1.0;
		}
		else
		{
			_Obstacles[pos].x = 1.0;
		}
	}
}

