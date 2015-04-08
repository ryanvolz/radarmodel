{{ name }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}


{% block methods_autosummary %}
{% if methods %}

..
   HACK -- the point here is that we don't want this to appear in the output since numpydoc takes care of that, but the autosummary should still generate the pages.
   .. autosummary::
      :toctree:
      {% for item in all_methods %}
      {%- if not item.startswith('_') or item in ['__call__'] %}
      {{ name }}.{{ item }}
      {%- endif -%}
      {%- endfor %}

{% endif %}
{% endblock %}

{% block attributes_autosummary %}
{% if attributes %}

..
   HACK -- the point here is that we don't want this to appear in the output since numpydoc takes care of that, but the autosummary should still generate the pages.
   .. autosummary::
      :toctree:
      {% for item in all_attributes %}
      {%- if not item.startswith('_') %}
      {{ name }}.{{ item }}
      {%- endif -%}
      {%- endfor %}

{% endif %}
{% endblock %}
