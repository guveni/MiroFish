import { ref, reactive } from 'vue'

export function useAgentLogs() {
  const agentLogs = ref([])
  const consoleLogs = ref([])
  const agentLogLine = ref(0)
  const consoleLogLine = ref(0)
  const expandedLogs = ref(new Set())
  const showRawResult = reactive({})
  const isConsoleCollapsed = ref(false)

  const toggleLogExpand = (log) => {
    const newSet = new Set(expandedLogs.value)
    if (newSet.has(log.timestamp)) {
      newSet.delete(log.timestamp)
    } else {
      newSet.add(log.timestamp)
    }
    expandedLogs.value = newSet
  }

  const isLogCollapsed = (log) => {
    if (['tool_call', 'tool_result', 'llm_response'].includes(log.action)) {
      return !expandedLogs.value.has(log.timestamp)
    }
    return false
  }

  const toggleRawResult = (timestamp, event, rightPanelRef) => {
    const button = event?.target
    const buttonRect = button?.getBoundingClientRect()
    const buttonTopBeforeToggle = buttonRect?.top
    
    showRawResult[timestamp] = !showRawResult[timestamp]
    
    if (button && buttonTopBeforeToggle !== undefined && rightPanelRef?.value) {
      setTimeout(() => {
        const newButtonRect = button.getBoundingClientRect()
        const buttonTopAfterToggle = newButtonRect.top
        const scrollDelta = buttonTopAfterToggle - buttonTopBeforeToggle
        rightPanelRef.value.scrollTop += scrollDelta
      }, 0)
    }
  }

  const toolConfig = {
    'insight_forge': {
      name: 'Deep Insight',
      color: 'purple',
      icon: 'lightbulb'
    },
    'panorama_search': {
      name: 'Panorama Search',
      color: 'blue',
      icon: 'globe'
    },
    'interview_agents': {
      name: 'Agent Interview',
      color: 'green',
      icon: 'users'
    },
    'quick_search': {
      name: 'Quick Search',
      color: 'orange',
      icon: 'zap'
    },
    'get_graph_statistics': {
      name: 'Graph Stats',
      color: 'cyan',
      icon: 'chart'
    },
    'get_entities_by_type': {
      name: 'Entity Query',
      color: 'pink',
      icon: 'database'
    }
  }

  const getToolDisplayName = (toolName) => {
    return toolConfig[toolName]?.name || toolName
  }

  const getToolColor = (toolName) => {
    return toolConfig[toolName]?.color || 'gray'
  }

  const getToolIcon = (toolName) => {
    return toolConfig[toolName]?.icon || 'tool'
  }

  // Parse helper functions
  const parseInsightForge = (text) => {
    const result = {
      query: '',
      simulationRequirement: '',
      stats: { facts: 0, entities: 0, relationships: 0 },
      subQueries: [],
      facts: [],
      entities: [],
      relations: []
    }
    
    try {
      const queryMatch = text.match(/(?:分析问题|Analysis query):\s*(.+?)(?:\n|$)/i)
      if (queryMatch) result.query = queryMatch[1].trim()
      
      const reqMatch = text.match(/(?:预测场景|Prediction scenario):\s*(.+?)(?:\n|$)/i)
      if (reqMatch) result.simulationRequirement = reqMatch[1].trim()
      
      const factMatch = text.match(/(?:相关预测事实|Relevant prediction facts):\s*(\d+)/i)
      const entityMatch = text.match(/(?:涉及Entities|Involved entities):\s*(\d+)/i)
      const relMatch = text.match(/(?:Relation Chains|Relationship chains):\s*(\d+)/i)
      if (factMatch) result.stats.facts = parseInt(factMatch[1])
      if (entityMatch) result.stats.entities = parseInt(entityMatch[1])
      if (relMatch) result.stats.relationships = parseInt(relMatch[1])
      
      const subQSection = text.match(/### (?:分析的子问题|Sub-queries Analyzed)\n([\s\S]*?)(?=\n###|$)/i)
      if (subQSection) {
        const lines = subQSection[1].split('\n').filter(l => l.match(/^\d+\./))
        result.subQueries = lines.map(l => l.replace(/^\d+\.\s*/, '').trim()).filter(Boolean)
      }
      
      const factsSection = text.match(/### (?:【Key Facts】|\[Key Facts\])[\s\S]*?\n([\s\S]*?)(?=\n###|$)/i)
      if (factsSection) {
        const lines = factsSection[1].split('\n').filter(l => l.match(/^\d+\./))
        result.facts = lines.map(l => {
          const match = l.match(/^\d+\.\s*"?(.+?)"?\s*$/)
          return match ? match[1].replace(/^"|"$/g, '').trim() : l.replace(/^\d+\.\s*/, '').trim()
        }).filter(Boolean)
      }
      
      const entitySection = text.match(/### (?:【Core Entities】|\[Core Entities\])\n([\s\S]*?)(?=\n###|$)/i)
      if (entitySection) {
        const entityText = entitySection[1]
        const entityBlocks = entityText.split(/\n(?=- \*\*)/).filter(b => b.trim().startsWith('- **'))
        result.entities = entityBlocks.map(block => {
          const nameMatch = block.match(/^-\s*\*\*(.+?)\*\*\s*\((.+?)\)/)
          const summaryMatch = block.match(/(?:摘要|Summary):\s*"?(.+?)"?(?:\n|$)/i)
          const relatedMatch = block.match(/(?:相关事实|Related facts):\s*(\d+)/i)
          return {
            name: nameMatch ? nameMatch[1].trim() : '',
            type: nameMatch ? nameMatch[2].trim() : '',
            summary: summaryMatch ? summaryMatch[1].trim() : '',
            relatedFactsCount: relatedMatch ? parseInt(relatedMatch[1]) : 0
          }
        }).filter(e => e.name)
      }
      
      const relSection = text.match(/### (?:【Relation Chains】|\[Relationship Chains\])\n([\s\S]*?)(?=\n###|$)/i)
      if (relSection) {
        const lines = relSection[1].split('\n').filter(l => l.trim().startsWith('-'))
        result.relations = lines.map(l => {
          const match = l.match(/^-\s*(.+?)\s*--\[(.+?)\]-->\s*(.+)$/)
          if (match) {
            return { source: match[1].trim(), relation: match[2].trim(), target: match[3].trim() }
          }
          return null
        }).filter(Boolean)
      }
    } catch (e) {
      console.warn('Parse insight_forge failed:', e)
    }
    
    return result
  }

  const parsePanorama = (text) => {
    const result = {
      query: '',
      stats: { nodes: 0, edges: 0, activeFacts: 0, historicalFacts: 0 },
      activeFacts: [],
      historicalFacts: [],
      entities: []
    }
    
    try {
      const queryMatch = text.match(/(?:查询|Query):\s*(.+?)(?:\n|$)/i)
      if (queryMatch) result.query = queryMatch[1].trim()
      
      const nodeMatch = text.match(/(?:返回Nodes数|Nodes returned):\s*(\d+)/i)
      const edgeMatch = text.match(/(?:Edges数|Edges returned):\s*(\d+)/i)
      const activeMatch = text.match(/(?:有效预测事实数|Active facts count):\s*(\d+)/i)
      const histMatch = text.match(/(?:历史演变事实数|Historical facts count):\s*(\d+)/i)
      if (nodeMatch) result.stats.nodes = parseInt(nodeMatch[1])
      if (edgeMatch) result.stats.edges = parseInt(edgeMatch[1])
      if (activeMatch) result.stats.activeFacts = parseInt(activeMatch[1])
      if (histMatch) result.stats.historicalFacts = parseInt(histMatch[1])
      
      const activeSection = text.match(/### (?:有效预测事实|Active simulation facts):\n([\s\S]*?)(?=\n###|$)/i)
      if (activeSection) {
        const lines = activeSection[1].split('\n').filter(l => l.match(/^\d+\./))
        result.activeFacts = lines.map(l => l.replace(/^\d+\.\s*/, '').trim()).filter(Boolean)
      }
      
      const histSection = text.match(/### (?:历史演变 facts|Historical evolution facts):\n([\s\S]*?)(?=\n###|$)/i)
      if (histSection) {
        const lines = histSection[1].split('\n').filter(l => l.match(/^\d+\./))
        result.historicalFacts = lines.map(l => l.replace(/^\d+\.\s*/, '').trim()).filter(Boolean)
      }
      
      const entitiesSection = text.match(/### (?:涉及的 Entities|Involved entities):\n([\s\S]*?)(?=\n###|$)/i)
      if (entitiesSection) {
        const lines = entitiesSection[1].split('\n').filter(l => l.trim().startsWith('-'))
        result.entities = lines.map(l => {
          const match = l.match(/^-\s*\*\*(.+?)\*\*\s*\((.+?)\)/)
          if (match) return { name: match[1].trim(), type: match[2].trim() }
          const simpleMatch = l.match(/^-\s*(.+)$/)
          if (simpleMatch) return { name: simpleMatch[1].trim(), type: '' }
          return null
        }).filter(Boolean)
      }
    } catch (e) {
      console.warn('Parse panorama_search failed:', e)
    }
    
    return result
  }

  const parseQuickSearch = (text) => {
    const result = {
      query: '',
      count: 0,
      facts: [],
      edges: [],
      nodes: []
    }
    
    try {
      const queryMatch = text.match(/(?:搜索查询|Search query):\s*(.+?)(?:\n|$)/i)
      if (queryMatch) result.query = queryMatch[1].trim()
      
      const countMatch = text.match(/(?:找到|Found)\s*(\d+)\s*(?:条|relevant)/i)
      if (countMatch) result.count = parseInt(countMatch[1])
      
      const factsSection = text.match(/### (?:相关事实|Relevant facts):\n([\s\S]*)$/i)
      if (factsSection) {
        const lines = factsSection[1].split('\n').filter(l => l.match(/^\d+\./))
        result.facts = lines.map(l => l.replace(/^\d+\.\s*/, '').trim()).filter(Boolean)
      }
      
      const edgesSection = text.match(/### (?:Related Edges|Relevant edges):\n([\s\S]*?)(?=\n###|$)/i)
      if (edgesSection) {
        const lines = edgesSection[1].split('\n').filter(l => l.trim().startsWith('-'))
        result.edges = lines.map(l => {
          const match = l.match(/^-\s*(.+?)\s*--\[(.+?)\]-->\s*(.+)$/)
          if (match) {
            return { source: match[1].trim(), relation: match[2].trim(), target: match[3].trim() }
          }
          return null
        }).filter(Boolean)
      }
      
      const nodesSection = text.match(/### (?:Related Nodes|Relevant nodes):\n([\s\S]*?)(?=\n###|$)/i)
      if (nodesSection) {
        const lines = nodesSection[1].split('\n').filter(l => l.trim().startsWith('-'))
        result.nodes = lines.map(l => {
          const match = l.match(/^-\s*\*\*(.+?)\*\*\s*\((.+?)\)/)
          if (match) return { name: match[1].trim(), type: match[2].trim() }
          const simpleMatch = l.match(/^-\s*(.+)$/)
          if (simpleMatch) return { name: simpleMatch[1].trim(), type: '' }
          return null
        }).filter(Boolean)
      }
    } catch (e) {
      console.warn('Parse quick_search failed:', e)
    }
    
    return result
  }

  const parseInterviewAgents = (text) => {
    const result = {
      topic: '',
      platforms: [],
      stats: { agentsCount: 0, repliesCount: 0, platformsCount: 0 },
      transcripts: []
    }
    
    try {
      const topicMatch = text.match(/(?:采访主题|Interview topic):\s*(.+?)(?:\n|$)/i)
      if (topicMatch) result.topic = topicMatch[1].trim()
      
      const platformMatch = text.match(/(?:采访平台|Platforms):\s*(.+?)(?:\n|$)/i)
      if (platformMatch) {
        result.platforms = platformMatch[1].split(',').map(p => p.trim())
        result.stats.platformsCount = result.platforms.length
      }
      
      const agentsCountMatch = text.match(/(?:采访Agents数|Agents interviewed):\s*(\d+)/i)
      if (agentsCountMatch) result.stats.agentsCount = parseInt(agentsCountMatch[1])
      
      const repliesCountMatch = text.match(/(?:采访回复数|Total replies collected):\s*(\d+)/i)
      if (repliesCountMatch) result.stats.repliesCount = parseInt(repliesCountMatch[1])
      
      const transSection = text.match(/### (?:采访实录|Interview Transcripts):\n([\s\S]*)$/i)
      if (transSection) {
        const transText = transSection[1]
        const agentBlocks = transText.split(/\n(?=- Agent \d+:)/).filter(b => b.trim().startsWith('- Agent '))
        result.transcripts = agentBlocks.map(block => {
          const titleMatch = block.match(/^-\s*Agent\s*(\d+):\s*(.+?)\s*\((.+?)\)/)
          const lines = block.split('\n')
          const qas = []
          let currentQa = null
          
          lines.forEach(line => {
            const platformLabelMatch = line.match(/^\s*(Twitter|Reddit):\s*$/i)
            if (platformLabelMatch) {
              if (currentQa) qas.push(currentQa)
              currentQa = { platform: platformLabelMatch[1], content: '' }
            } else if (currentQa) {
              if (line.trim().startsWith('>')) {
                currentQa.content += (currentQa.content ? '\n' : '') + line.replace(/^\s*>\s*/, '').trim()
              }
            }
          })
          if (currentQa) qas.push(currentQa)
          
          return {
            agentId: titleMatch ? parseInt(titleMatch[1]) : 0,
            name: titleMatch ? titleMatch[2].trim() : '',
            platform: titleMatch ? titleMatch[3].trim() : '',
            qas: qas
          }
        }).filter(t => t.name)
      }
    } catch (e) {
      console.warn('Parse interview_agents failed:', e)
    }
    
    return result
  }

  return {
    agentLogs,
    consoleLogs,
    agentLogLine,
    consoleLogLine,
    expandedLogs,
    showRawResult,
    isConsoleCollapsed,
    toggleLogExpand,
    isLogCollapsed,
    toggleRawResult,
    getToolDisplayName,
    getToolColor,
    getToolIcon,
    parseInsightForge,
    parsePanorama,
    parseQuickSearch,
    parseInterviewAgents
  }
}
