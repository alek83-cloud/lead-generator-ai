from crewai import Agent, Crew, Process, Task, LLM  # ⬅️ add LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
import os

load_dotenv()

# Tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Structured output schema
class LeadOutput(BaseModel):
    company_name: Optional[str] = Field(description="The name of the company")
    annual_revenue: Optional[str] = Field(description="Annual revenue of the company")
    location: Optional[Dict[str, str]] = Field(description="Location with city and country fields")
    website_url: Optional[str] = Field(description="Company website URL")
    review: Optional[str] = Field(description="Description of what the company does")
    num_employees: Optional[int] = Field(description="Number of employees")
    key_decision_makers: Optional[List[Dict[str, str]]] = Field(description="List of key people with their LinkedIn profiles")
    score: Optional[int] = Field(description="Fit score on a scale of 1-10")

@CrewBase
class LeadGenerator():
    """LeadGenerator crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # ⬇️ NEW: constructor to choose provider/model once for the whole crew
    def __init__(self, provider: str = "openai", model: Optional[str] = None, temperature: float = 0.2, max_tokens: Optional[int] = None):
        provider = (provider or "openai").lower().strip()
        # sensible defaults
        if not model:
            model = "openai/gpt-4o" if provider == "openai" else "gemini/gemini-1.5-pro"
        # allow plain names like "gpt-4o" or "gemini-1.5-pro"
        if "/" not in model:
            prefix = "openai" if provider == "openai" else "gemini"
            model = f"{prefix}/{model}"
        # one LLM for all agents
        self.llm = LLM(model=model, temperature=temperature, max_tokens=max_tokens)
        # env keys expected by CrewAI:
        # OPENAI_API_KEY for openai/*, GEMINI_API_KEY for gemini/*  (see docs)  # noqa

    @agent
    def lead_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['lead_generator'],
            tools=[search_tool, scrape_tool],
            verbose=True,
            llm=self.llm  # ⬅️ override YAML/defaults with our chosen LLM
        )

    @agent
    def contact_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['contact_agent'],
            tools=[search_tool, scrape_tool],
            verbose=True,
            llm=self.llm
        )

    @agent 
    def lead_qualifier(self) -> Agent:
        return Agent(
            config=self.agents_config['lead_qualifier'],
            verbose=True,
            llm=self.llm
        )

    @agent
    def sales_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['sales_manager'],
            tools=[],
            verbose=True,
            llm=self.llm
        )

    @task
    def lead_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config['lead_generation_task'],
            output_pydantic=LeadOutput
        )

    @task
    def contact_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['contact_research_task'],
            context=[self.lead_generation_task()],
        )

    @task
    def lead_qualification_task(self) -> Task:
        return Task(
            config=self.tasks_config['lead_qualification_task'],
            context=[self.lead_generation_task(), self.contact_research_task()],
            output_pydantic=LeadOutput,
        )

    @task
    def sales_management_task(self) -> Task:
        return Task(
            config=self.tasks_config['sales_management_task'],
            context=[self.lead_generation_task(), self.lead_qualification_task(), self.contact_research_task()],
            output_pydantic=LeadOutput
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            usage_metrics={}  # keep your usage collection
            # If you use a manager/planner, you can also pass an LLM object:
            # manager_llm=self.llm
        )
