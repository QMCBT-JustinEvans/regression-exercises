USE zillow;
SELECT * FROM properties_2017
LEFT JOIN airconditioningtype USING (airconditioningtypeid)
LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
LEFT JOIN buildingclasstype USING (buildingclasstypeid)
LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
LEFT JOIN propertylandusetype USING (propertylandusetypeid)
LEFT JOIN storytype USING (storytypeid)
LEFT JOIN typeconstructiontype USING (typeconstructiontypeid);

