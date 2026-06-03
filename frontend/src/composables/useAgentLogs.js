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

  /** Single-line field values only; avoids capturing markdown headings on the next line. */
  const extractLineValue = (text, pattern) => {
    const match = text.match(pattern)
    if (!match) return ''
    const value = match[1].trim()
    return value.startsWith('#') ? '' : value
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
      result.query = extractLineValue(text, /(?:分析问题|Analysis query):\s*([^\n]+)/i)
      result.simulationRequirement = extractLineValue(text, /(?:预测场景|Prediction scenario):\s*([^\n]+)/i)
      
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
      result.query = extractLineValue(text, /(?:查询|Query):\s*([^\n]+)/i)

      const nodesMatch = text.match(/(?:总节点数|Total nodes):\s*(\d+)/i)
      const edgesMatch = text.match(/(?:总边数|Total edges):\s*(\d+)/i)
      const activeMatch = text.match(/(?:当前有效事实|Current valid facts):\s*(\d+)/i)
      const histMatch = text.match(/(?:历史\/过期事实|Historical\/expired facts):\s*(\d+)/i)
      if (nodesMatch) result.stats.nodes = parseInt(nodesMatch[1], 10)
      if (edgesMatch) result.stats.edges = parseInt(edgesMatch[1], 10)
      if (activeMatch) result.stats.activeFacts = parseInt(activeMatch[1], 10)
      if (histMatch) result.stats.historicalFacts = parseInt(histMatch[1], 10)

      const activeSection = text.match(/### (?:【当前有效事实】|\[Current Valid Facts\])[\s\S]*?\n([\s\S]*?)(?=\n###|$)/i)
      if (activeSection) {
        const lines = activeSection[1].split('\n').filter(l => l.match(/^\d+\./))
        result.activeFacts = lines.map(l => {
          const factText = l.replace(/^\d+\.\s*/, '').replace(/^"|"$/g, '').trim()
          return factText
        }).filter(Boolean)
      }

      const histSection = text.match(/### (?:【历史\/过期事实】|\[Historical\/Expired Facts\])[\s\S]*?\n([\s\S]*?)(?=\n###|$)/i)
      if (histSection) {
        const lines = histSection[1].split('\n').filter(l => l.match(/^\d+\./))
        result.historicalFacts = lines.map(l => {
          const factText = l.replace(/^\d+\.\s*/, '').replace(/^"|"$/g, '').trim()
          return factText
        }).filter(Boolean)
      }

      const entitySection = text.match(/### (?:【涉及Entities】|\[Involved Entities\])\n([\s\S]*?)(?=\n###|$)/i)
      if (entitySection) {
        const lines = entitySection[1].split('\n').filter(l => l.trim().startsWith('-'))
        result.entities = lines.map(l => {
          const match = l.match(/^-\s*\*\*(.+?)\*\*\s*\((.+?)\)/)
          if (match) return { name: match[1].trim(), type: match[2].trim() }
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
      result.query = extractLineValue(text, /(?:搜索查询|Search query):\s*([^\n]+)/i)
      
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
    agentCount: '',
    successCount: 0,
    totalCount: 0,
    selectionReason: '',
    interviews: [],
    summary: ''
  }
  
  try {
    // Extract interview topic
    const topicMatch = text.match(/\*\*(?:采访主题|Interview Topic):\*\*\s*(.+?)(?:\n|$)/i)
    if (topicMatch) result.topic = topicMatch[1].trim()
    
    // Extract interview count
    const countMatch = text.match(/\*\*(?:采访人数|Interview Count):\*\*\s*(\d+)\s*\/\s*(\d+)/i)
    if (countMatch) {
      result.successCount = parseInt(countMatch[1])
      result.totalCount = parseInt(countMatch[2])
      result.agentCount = `${countMatch[1]} / ${countMatch[2]}`
    }
    
    // Extract reasoning for selecting interview subjects
    const reasonMatch = text.match(/### (?:采访对象选择理由|Selection Reasoning)\n([\s\S]*?)(?=\n---\n|\n### 采访实录|\n### Interview Transcript)/i)
    if (reasonMatch) {
      result.selectionReason = reasonMatch[1].trim()
    }
    
    // Parse the selection reasoning for each person
    const parseIndividualReasons = (reasonText) => {
      const reasons = {}
      if (!reasonText) return reasons
      
      const lines = reasonText.split(/\n+/)
      let currentName = null
      let currentReason = []
      
      for (const line of lines) {
        let headerMatch = null
        let name = null
        let reasonStart = null
        
        // Format 1: Number. **Name (index=X)**: Reason
        headerMatch = line.match(/^\d+\.\s*\*\*([^*（(]+)(?:[（(]index\s*=?\s*\d+[)）])?\*\*[：:]\s*(.*)/)
        if (headerMatch) {
          name = headerMatch[1].trim()
          reasonStart = headerMatch[2]
        }
        
        // Format 2: - Select Name (index X): Reason
        if (!headerMatch) {
          headerMatch = line.match(/^-\s*选择([^（(]+)(?:[（(]index\s*=?\s*\d+[)）])?[：:]\s*(.*)/)
          if (headerMatch) {
            name = headerMatch[1].trim()
            reasonStart = headerMatch[2]
          }
        }
        
        // Format 3: - **Name (index X)**: Reason
        if (!headerMatch) {
          headerMatch = line.match(/^-\s*\*\*([^*（(]+)(?:[（(]index\s*=?\s*\d+[)）])?\*\*[：:]\s*(.*)/)
          if (headerMatch) {
            name = headerMatch[1].trim()
            reasonStart = headerMatch[2]
          }
        }
        
        if (name) {
          if (currentName && currentReason.length > 0) {
            reasons[currentName] = currentReason.join(' ').trim()
          }
          currentName = name
          currentReason = reasonStart ? [reasonStart.trim()] : []
        } else if (currentName && line.trim() && !line.match(/^未选|^综上|^最终选择/)) {
          currentReason.push(line.trim())
        }
      }
      
      if (currentName && currentReason.length > 0) {
        reasons[currentName] = currentReason.join(' ').trim()
      }
      
      return reasons
    }
    
    const individualReasons = parseIndividualReasons(result.selectionReason)
    
    // Extract each interview transcript
    const interviewBlocks = text.split(/#### (?:采访|Interview) #\d+:/i).slice(1)
    
    interviewBlocks.forEach((block, index) => {
      const interview = {
        num: index + 1,
        title: '',
        name: '',
        role: '',
        bio: '',
        selectionReason: '',
        questions: [],
        twitterAnswer: '',
        redditAnswer: '',
        quotes: []
      }
      
      // Extract title
      const titleMatch = block.match(/^(.+?)\n/)
      if (titleMatch) interview.title = titleMatch[1].trim()
      
      // Extract name and role
      const nameRoleMatch = block.match(/\*\*(.+?)\*\*\s*\((.+?)\)/)
      if (nameRoleMatch) {
        interview.name = nameRoleMatch[1].trim()
        interview.role = nameRoleMatch[2].trim()
        interview.selectionReason = individualReasons[interview.name] || ''
      }
      
      // Extract bio
      const bioMatch = block.match(/_(?:简介|Bio):\s*([\s\S]*?)_\n/i)
      if (bioMatch) {
        interview.bio = bioMatch[1].trim().replace(/\.\.\.$/, '...')
      }
      
      // Extract question list
      const qMatch = block.match(/\*\*Q:\*\*\s*([\s\S]*?)(?=\n\n\*\*A:\*\*|\*\*A:\*\*)/)
      if (qMatch) {
        const qText = qMatch[1].trim()
        const questions = qText.split(/\n\d+\.\s+/).filter(q => q.trim())
        if (questions.length > 0) {
          const firstQ = qText.match(/^1\.\s+(.+)/)
          if (firstQ) {
            interview.questions = [firstQ[1].trim(), ...questions.slice(1).map(q => q.trim())]
          } else {
            interview.questions = questions.map(q => q.trim())
          }
        }
      }
      
      // Extract answers - split into Twitter and Reddit
      const answerMatch = block.match(/\*\*A:\*\*\s*([\s\S]*?)(?=\*\*(?:关键引言|Key Quotes)|$)/i)
      if (answerMatch) {
        const answerText = answerMatch[1].trim()
        
        // Separate Twitter and Reddit answers
        const twitterMatch = answerText.match(/(?:【Twitter平台回答】|\[Twitter Platform Response\])\n?([\s\S]*?)(?=(?:【Reddit平台回答】|\[Reddit Platform Response\])|$)/i)
        const redditMatch = answerText.match(/(?:【Reddit平台回答】|\[Reddit Platform Response\])\n?([\s\S]*?)$/i)
        
        if (twitterMatch) {
          interview.twitterAnswer = twitterMatch[1].trim()
        }
        if (redditMatch) {
          interview.redditAnswer = redditMatch[1].trim()
        }
        
        // Platform fallback logic
        if (!twitterMatch && redditMatch) {
          if (interview.redditAnswer && interview.redditAnswer !== '（该平台未获得回复）' && interview.redditAnswer !== ' (No response received on this platform)') {
            interview.twitterAnswer = interview.redditAnswer
          }
        } else if (twitterMatch && !redditMatch) {
          if (interview.twitterAnswer && interview.twitterAnswer !== '（该平台未获得回复）' && interview.twitterAnswer !== ' (No response received on this platform)') {
            interview.redditAnswer = interview.twitterAnswer
          }
        } else if (!twitterMatch && !redditMatch) {
          interview.twitterAnswer = answerText
        }
      }
      
      // Extract key quotes
      const quotesMatch = block.match(/\*\*(?:关键引言|Key Quotes):\*\*\n([\s\S]*?)(?=\n---|\n####|$)/i)
      if (quotesMatch) {
        const quotesText = quotesMatch[1]
        let quoteMatches = quotesText.match(/> "([^"]+)"/g)
        if (!quoteMatches) {
          quoteMatches = quotesText.match(/> [\u201C""]([^\u201D""]+)[\u201D""]/g)
        }
        if (quoteMatches) {
          interview.quotes = quoteMatches
            .map(q => q.replace(/^> [\u201C""]|[\u201D""]$/g, '').trim())
            .filter(q => q)
        }
      }
      
      if (interview.name || interview.title) {
        result.interviews.push(interview)
      }
    })
    
    // Extract interview summary
    const summaryMatch = text.match(/### (?:采访摘要与核心观点|Interview Summary & Core Perspectives)\n([\s\S]*?)$/i)
    if (summaryMatch) {
      result.summary = summaryMatch[1].trim()
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
