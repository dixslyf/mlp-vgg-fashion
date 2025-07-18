function Div(el)
   if not el.classes:includes("cell") then
      return el
   end

   local cell_code = nil
   local output_children = {}
   for _, child in ipairs(el.content) do
      if child.t == "CodeBlock" and child.classes:includes("cell-code") then
         cell_code = child
      else
         -- Everything else is assumed to be part of the cell's output.
         table.insert(output_children, child)
      end
   end

   if #output_children <= 0 then
      return el
   end

   local new_content = {}

   if cell_code ~= nil then
      table.insert(new_content, cell_code)
   end

   if #output_children > 0 then
      local wrapper_div = pandoc.Div(output_children, { class = "cell-output-container" })
      table.insert(new_content, wrapper_div)
      assert((1 + #wrapper_div.content) == #el.content, "Unexpected length mismatch between old and new output content")
   else
      assert(#new_content == #el.content, "Unexpected length mismatch between old and new output content")
   end

   el.content = new_content
   return el
end
