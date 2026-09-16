const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json()
}

export function createTask({ task_type, user_id, input_data }) {
  return request('/tasks', {
    method: 'POST',
    body: JSON.stringify({ task_type, user_id, input_data }),
  })
}

export function getTask(taskId) {
  return request(`/tasks/${taskId}`)
}

export function getUserStyle(userId) {
  return request(`/users/${userId}/style`)
}

export function getMetrics() {
  return request('/metrics')
}

export { BASE_URL }
